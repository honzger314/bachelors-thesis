import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):

    """
    CNN for CIFAR-10.

    Input:
        3 x 32 x 32 RGB image

    Output:
        10 classes
    """


    def __init__(self):

        super().__init__()


        self.conv1 = nn.Conv2d(
            3,
            32,
            kernel_size=3,
            padding=1
        )


        self.conv2 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1
        )


        self.conv3 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1
        )


        self.pool = nn.MaxPool2d(
            2,
            2
        )


        self.fc1 = nn.Linear(
            128 * 4 * 4,
            256
        )


        self.fc2 = nn.Linear(
            256,
            10
        )


    def forward(self, x):

        # 32x32
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        # 16x16
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # 8x8
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        # 4x4
        x = torch.flatten(
            x,
            start_dim=1
        )


        x = F.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x



def create_model():

    return SimpleCNN()