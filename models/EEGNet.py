import torch.nn as nn


# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self, activation="elu", alpha=1.0, dropout=0.25):
        super(EEGNet, self).__init__()

        # Create a separate activation module for each convolution block.
        def make_activation():
            if activation == "elu":
                return nn.ELU(alpha=alpha)
            if activation == "relu":
                return nn.ReLU()
            if activation == "leaky_relu":
                return nn.LeakyReLU(negative_slope=0.01)
            raise ValueError("activation must be 'elu', 'relu', or 'leaky_relu'")

        # conv2d + batchnorm
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            make_activation(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
            nn.Dropout(p=dropout)
        )

        self.separableConv = nn.Sequential(
            # nn.Conv2d(32, 32, kernel_size=(1, 15),padding=(0, 7), stride=(1, 1), padding=(0, 7),groups=32, bias=False,),
            nn.Conv2d(32, 32, kernel_size=(1, 15),padding=(0, 7), groups=32, bias=False,),
            nn.Conv2d(32, 32, kernel_size=(1, 1), bias=False,),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            make_activation(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=dropout)
        )

        self.classify = nn.Sequential(
            nn.Linear(736, 2,  bias=True)
        )

    def forward(self, x):
        x = self.firstconv(x)
        x = self.depthwiseConv(x)
        x = self.separableConv(x)
        x = x.flatten(start_dim=1)
        return self.classify(x)    


# (Optional) implement DeepConvNet model
class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()
        # reshape the input to (1, 2, 750)
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 25, kernel_size=(1, 5)),
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.secondconv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.thirdconv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )
        self.fourthconv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1),
            nn.ELU(alpha=1.0),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(p=0.5)
        )


        self.classify = nn.Linear(200 * 43, 2)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.secondconv(x)
        x = self.thirdconv(x)
        x = self.fourthconv(x)
        x = x.flatten(start_dim=1)
        return self.classify(x)
