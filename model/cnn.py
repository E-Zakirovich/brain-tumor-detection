import torch
import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, num_classes = 4):
        # I need to inherit Neural Network from torch.nn to make the network
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,  # RGB image
            out_channels=32,  # learn 32 feature maps
            kernel_size=3, # 3x3 filter
            padding=1 # keep image size unchanged
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )

        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.relu = nn.ReLU(inplace=True)

        self.flatten_dim = 256 * 8 * 8

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(
            self.flatten_dim,
            512
        )

        self.fc2 = nn.Linear(
            512,
            num_classes
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x