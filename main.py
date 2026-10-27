import copy
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


def train(model, loader, criterion, optimizer, args):
    best_acc = 0.0
    best_wts = None
    avg_acc_list = []
    test_acc_list = []
    test_loss_list = []
    avg_loss_list = []
    for epoch in range(1, args.num_epochs+1):
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
                if args.mode == "E" and args.type in (9, 10):
                    max_norm = 0.25 if args.type == 9 else 1.0
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
    parser.add_argument("-lr", type=float, default=0.01)
    
    parser.add_argument("-ks", type=int, default=51, help="kernel size for convolutional layers")

    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-type", type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], help="0 for elu, 1 for relu, 2 for leaky relu, 3 for elu(alpha=0.5), 4 for elu(dropout=0.5), 5 for relu(dropout=0.5), 6 for relu(dropout=0.1), 7 for relu(dropout=0.35), 8 for relu(dropout=0.35) with custom kernel size, 9 for relu(dropout=0.35, kernel=51) with classifier max-norm=0.25, 10 for relu(dropout=0.35, kernel=51) with classifier max-norm=1.0")

    args = parser.parse_args()

    setseed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model_name = args.mode+str(args.type)+"_"+str(args.seed)
    folder_name = model_name + '_' + datetime.now().strftime("%d%H%M%S")
    # check the results folder exist or not, if not, create it
    if not os.path.exists(RESULT_DIR+VERSION):
        os.makedirs(RESULT_DIR+VERSION)
    if not os.path.exists(RESULT_DIR + VERSION + f'/{folder_name}'):
        os.makedirs(RESULT_DIR + VERSION + f'/{folder_name}')

    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    train_dataset = BCIDataset(train_data, train_label)
    test_dataset = BCIDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # TODO write EEGNet yourself
    if args.mode == "E":
        match args.type:
            case 0:
                model = EEGNet(activation="elu", alpha=1.0, dropout=0.25)
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
            case _:
                raise ValueError("Invalid type. Choose an integer from 0 to 10.")
    elif args.mode == "D":
        model = DeepConvNet()
    else:
        raise ValueError("Invalid mode. Choose either 'E' for EEGNet or 'D' for DeepConvNet.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)

    model.to(device)
    criterion.to(device)

    train_acc_list, train_loss_list, test_acc_list, test_loss_list = train(model, train_loader, criterion, optimizer, args)

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
        
