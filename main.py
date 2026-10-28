import copy
import json
from unittest import case
import torch
import argparse
import dataloader
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
import torch.optim as optim
import matplotlib.pyplot as plt
from models.EEGNet import DeepConvNet, EEGNet
from torchsummary import summary
from matplotlib.ticker import MaxNLocator
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import os
import random

RESULT_DIR = './results'
WEIGHTS_DIR = './weights'
VERSION = '_2sep_reproduce'
folder_name = ""
model_name = ""


def setseed(seed=42):
    """Fix random seeds and cuDNN settings for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class BCIDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = torch.tensor(self.data[index,...], dtype=torch.float32)
        label = torch.tensor(self.label[index], dtype=torch.int64)
        return data, label

    def __len__(self):
        return self.data.shape[0]

def plot_train_acc(train_acc_list, epochs):
    # TODO plot training accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_acc_list, label='Training Accuracy', marker='o')

    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(RESULT_DIR + VERSION + f'/{folder_name}/accuracy_curve.png')
    # save the data of both acc
    # so that it can reproduce the plot later
    np.save(RESULT_DIR + VERSION + f'/{folder_name}/train_acc.npy', np.array(train_acc_list))

def plot_train_loss(train_loss_list, epochs):
    # TODO plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_loss_list, label='Training Loss', marker='o')

    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(RESULT_DIR + VERSION + f'/{folder_name}/loss_curve.png')
    # save the data of both acc
    # so that it can reproduce the plot later
    np.save(RESULT_DIR + VERSION + f'/{folder_name}/train_loss.npy', np.array(train_loss_list))

def plot_test_acc(test_acc_list, epochs):
    # TODO plot testing loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), test_acc_list, label='Testing Accuracy', marker='o')

    plt.title('Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig(RESULT_DIR + VERSION + f'/{folder_name}/test_accuracy_curve.png')
    # save the data of both acc
    # so that it can reproduce the plot later
    np.save(RESULT_DIR + VERSION + f'/{folder_name}/test_acc.npy', np.array(test_acc_list))

def plot_test_loss(test_loss_list, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), test_loss_list, label='Testing Loss', marker='o')
    plt.title('Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(RESULT_DIR + VERSION + f'/{folder_name}/test_loss_curve.png')
    plt.close()
    np.save(RESULT_DIR + VERSION + f'/{folder_name}/test_loss.npy', np.array(test_loss_list))


def train(model, loader, criterion, optimizer, args, scheduler=None):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    test_loss_list = []
    avg_loss_list = []
    lr_list = []
    for epoch in range(1, args.num_epochs+1):
        if scheduler is not None:
            # Record the rate used for this epoch, before the end-of-epoch step.
            lr_list.append(optimizer.param_groups[0]["lr"])
            print(f'Learning rate: {lr_list[-1]:.8f}')
        model.train()
        with torch.set_grad_enabled(True):
            avg_acc = 0.0
            avg_loss = 0.0 
            for i, data in enumerate(tqdm(loader), 0):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                if args.mode == "E" and args.type in (0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
                    max_norm = {0: 1.0, 9: 0.25, 10: 1.0, 11: 2.0, 12: 1.0, 13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0, 21: 1.0, 22: 1.0, 23: 1.0, 24: 1.0, 25: 1.0, 26: 1.0, 27: 1.0, 28: 1.0, 29: 1.0, 30: 1.0, 31: 1.0, 32: 1.0}[args.type]
                    # Project each classifier weight vector onto the L2 ball.
                    # Leave the bias and weights already within the limit unchanged.
                    with torch.no_grad():
                        weight = model.classify[0].weight
                        norms = weight.norm(p=2, dim=1, keepdim=True)
                        weight.mul_(max_norm / norms.clamp(min=max_norm))
                # avg_loss += loss.item() * batch_size
                avg_loss += loss.item() * inputs.size(0)  
                _, pred = torch.max(outputs.data, 1)
                avg_acc += pred.eq(labels).cpu().sum().item()

            avg_loss /= len(loader.dataset)
            avg_loss_list.append(avg_loss)
            avg_acc = (avg_acc / len(loader.dataset)) * 100
            avg_acc_list.append(avg_acc)
            print(f'Epoch: {epoch}')
            print(f'Loss: {avg_loss}')
            print(f'Training Acc. (%): {avg_acc:3.2f}%')

        test_acc, test_loss = test(model, test_loader, criterion)
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)
        if test_acc > best_acc:
            best_acc = test_acc
            best_wts = copy.deepcopy(model.state_dict())
        print(f'Test Acc. (%): {test_acc:3.2f}%')
        print(f'Test Loss: {test_loss:.6f}')
        if scheduler is not None:
            scheduler.step()
    if scheduler is not None:
        np.save(RESULT_DIR + VERSION + f'/{folder_name}/lr.npy', np.array(lr_list))
    if not os.path.exists(WEIGHTS_DIR+VERSION):
        os.makedirs(WEIGHTS_DIR+VERSION)
    torch.save(best_wts, WEIGHTS_DIR + VERSION + '/' + folder_name + '_best.pt')
    return avg_acc_list, avg_loss_list, test_acc_list, test_loss_list


def test(model, loader, criterion):
    avg_acc = 0.0
    avg_loss = 0.0
    model.eval()
    with torch.set_grad_enabled(False):
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            avg_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(outputs, 1)
            for i in range(len(labels)):
                if int(pred[i]) == int(labels[i]):
                    avg_acc += 1

        avg_acc = (avg_acc / len(loader.dataset)) * 100
        avg_loss /= len(loader.dataset)

    return avg_acc, avg_loss



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, default="E", choices=["E", "D"])
    parser.add_argument("-num_epochs", type=int, default=150)
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lr", type=float, default=None,
                        help="initial learning rate (default: 0.005 for E0 and E25-E32, 0.001 for E24, 0.003 for E23, 0.01 otherwise)")
    
    parser.add_argument("-ks", type=int, default=51, help="kernel size for convolutional layers")

    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("--summary-only", action="store_true",
                        help="print model architecture and summary, without training")
    parser.add_argument("-type", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], help="0 for assignment spec convolution with case 25 settings, 1 for relu, 2 for leaky relu, 3 for elu(alpha=0.5), 4 for elu(dropout=0.5), 5 for relu(dropout=0.5), 6 for relu(dropout=0.1), 7 for relu(dropout=0.35), 8 for relu(dropout=0.35) with custom kernel size, 9 for relu(dropout=0.35, kernel=51) with classifier max-norm=0.25, 10 for relu(dropout=0.35, kernel=51) with classifier max-norm=1.0, 11 for relu(dropout=0.35, kernel=51) with classifier max-norm=2.0, 12 for relu(dropout=0.25, kernel=51) with classifier max-norm=1.0, 13 for relu(dropout=0.35, kernel=51, F1=8, D=2, F2=16) with classifier max-norm=1.0, 14 for relu(dropout=0.35, kernel=51, BN momentum=0.01) with classifier max-norm=1.0, 15 for relu(dropout=0.35, kernel=51, BN momentum=0.3) with classifier max-norm=1.0, 16 for relu(dropout=0.35, kernel=51, BN momentum=0.5) with classifier max-norm=1.0, 17 for relu(dropout=0.35, kernel=51, BN momentum=0.7) with classifier max-norm=1.0, 18 for leaky_relu(slope=0.01, dropout=0.35, kernel=51, BN momentum=0.7) with classifier max-norm=1.0, 19 for leaky_relu(slope=0.1, dropout=0.35, kernel=51, BN momentum=0.7) with classifier max-norm=1.0, 20 for leaky_relu(slope=0.01, dropout=0.35, kernel=31, BN momentum=0.7) with classifier max-norm=1.0, 21 for leaky_relu(slope=0.01, dropout=0.25, kernel=31, BN momentum=0.7) with classifier max-norm=1.0, 22 for case 20 architecture with cosine LR decay to 0.0001, 23 for case 20 architecture with cosine LR decay and initial lr=0.003, 24 for case 20 architecture with cosine LR decay and initial lr=0.001, 25 for case 20 architecture with cosine LR decay and initial lr=0.005, 26 for case 25 with offline 4-30 Hz FIR preprocessing (125 Hz, 101 taps), 27 for case 25 with 4 Hz highpass FIR, 28 for case 25 with 30 Hz lowpass FIR, 29 for case 25 with ReLU activation, 30 for case 25 with temporal kernel=51, 31 for case 25 with temporal kernel=63, 32 for case 25 with temporal kernel=15 and 4 Hz highpass FIR")

    args = parser.parse_args()
    if args.lr is None:
        args.lr = {0: 0.005, 23: 0.003, 24: 0.001, 25: 0.005, 26: 0.005, 27: 0.005, 28: 0.005, 29: 0.005, 30: 0.005, 31: 0.005, 32: 0.005}.get(args.type, 0.01) if args.mode == "E" else 0.01

    setseed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_name = args.mode+str(args.type)+"_"+str(args.seed)
    folder_name = model_name + '_' + datetime.now().strftime("%d%H%M%S")
    # check the results folder exist or not, if not, create it
    if not args.summary_only:
        os.makedirs(RESULT_DIR + VERSION + f'/{folder_name}', exist_ok=True)

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    if args.mode == "E" and args.type == 26:
        train_data = dataloader.bandpass_eeg(train_data)
        test_data = dataloader.bandpass_eeg(test_data)
        if not args.summary_only:
            with open(RESULT_DIR + VERSION + f'/{folder_name}/preprocessing.txt', "w") as f:
                f.write("FIR bandpass: 4-30 Hz; assumed fs=125 Hz; 101 taps; Hamming window; "
                        "unit gain at 17 Hz; reflect padding; centered offline convolution; "
                        "per trial and channel, after existing NaN filling.\n")
    if args.mode == "E" and args.type in (27, 28, 32):
        kind, cutoff = ("highpass", 4.0) if args.type in (27, 32) else ("lowpass", 30.0)
        train_data = dataloader.single_cutoff_eeg(train_data, cutoff, kind)
        test_data = dataloader.single_cutoff_eeg(test_data, cutoff, kind)
        reference = "Nyquist" if kind == "highpass" else "DC"
        if not args.summary_only:
            with open(RESULT_DIR + VERSION + f'/{folder_name}/preprocessing.txt', "w") as f:
                f.write(f"FIR {kind}: cutoff={cutoff} Hz; assumed fs=125 Hz; 101 taps; "
                        f"Hamming window; unit gain at {reference}; reflect padding; "
                        "centered offline convolution; per trial and channel, after existing NaN filling.\n")
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # TODO write EEGNet yourself
    if args.mode == "E":
        match args.type:
            case 0:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=31,
                               bn_momentum=0.7, leaky_slope=0.01, separable_mode="spec")
            case 1:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.25)
            case 2:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.25)
            case 3:
                model = EEGNet(activation="elu", alpha=0.5, dropout=0.25)
            case 4:
                model = EEGNet(activation="elu", alpha=1.0, dropout=0.5)
            case 5:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.5)
            case 6:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.1)
            case 7:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35)
            case 8:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=args.ks)
            case 9:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=51)
            case 10:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=51)
            case 11:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=51)
            case 12:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.25, ks=51)
            case 13:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=51,
                               f1=8, d=2, f2=16)
            case 14:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=51,
                               bn_momentum=0.01)
            case 15:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=51,
                               bn_momentum=0.3)
            case 16:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=51,
                               bn_momentum=0.5)
            case 17:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=51,
                               bn_momentum=0.7)
            case 18:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=51,
                               bn_momentum=0.7)
            case 19:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=51,
                               bn_momentum=0.7, leaky_slope=0.1)
            case 20:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=31,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 21:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.25, ks=31,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 22:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=31,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 23:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=31,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 24:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=31,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 25:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=31,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 26:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=31,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 27 | 28:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=31,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 29:
                model = EEGNet(activation="relu", alpha=1.0, dropout=0.35, ks=31,
                               bn_momentum=0.7)
            case 30:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=51,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 31:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=63,
                               bn_momentum=0.7, leaky_slope=0.01)
            case 32:
                model = EEGNet(activation="leaky_relu", alpha=1.0, dropout=0.35, ks=15,
                               bn_momentum=0.7, leaky_slope=0.01)
            case _:
                raise ValueError("Invalid type. Choose an integer from 0 to 32.")
    elif args.mode == "D":
        model = DeepConvNet()
    else:
        raise ValueError("Invalid mode. Choose either 'E' for EEGNet or 'D' for DeepConvNet.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    model.to(device)
    # criterion.to(device)
    # print(model)
    if args.summary_only:
        print(model)
        model.eval()
        summary(model, input_size=(1, 2, 750), device=device.type)
        raise SystemExit(0)
    scheduler = None
    if args.mode == "E" and args.type in (0, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs, eta_min=0.0001
        )
    # Record the actual architecture because case 0 now has a new meaning.
    with open(RESULT_DIR + VERSION + f'/{folder_name}/run_config.json', "w") as f:
        json.dump({
            "args": vars(args),
            "architecture": ("spec" if args.type == 0 else "2sep") if args.mode == "E" else "DeepConvNet",
            "model": str(model),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "scheduler": None if scheduler is None else {
                "name": "CosineAnnealingLR",
                "T_max": args.num_epochs,
                "eta_min": 0.0001,
            },
        }, f, indent=2)
    train_acc_list, train_loss_list, test_acc_list, test_loss_list = train(
        model, train_loader, criterion, optimizer, args, scheduler=scheduler
    )

    plot_train_acc(train_acc_list, args.num_epochs)
    plot_train_loss(train_loss_list, args.num_epochs)
    plot_test_acc(test_acc_list, args.num_epochs)
    plot_test_loss(test_loss_list, args.num_epochs)

    # test with the best model
    best_model = copy.deepcopy(model)
    best_model.load_state_dict(torch.load(WEIGHTS_DIR + VERSION + '/' + folder_name + '_best.pt'))
    best_model.to(device)
    best_test_acc, best_test_loss = test(best_model, test_loader, criterion)
    # save the best test accuracy and loss to a text file
    with open(RESULT_DIR + VERSION + f'/{folder_name}/best_test_results.txt', 'w') as f:
        f.write(f'Best Test Accuracy: {best_test_acc:.2f}%\n')
        f.write(f'The loss of the best model: {best_test_loss:.6f}\n')
        
