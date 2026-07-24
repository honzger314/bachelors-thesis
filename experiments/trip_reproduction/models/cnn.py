import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    Small CNN for MNIST/FashionMNIST experiments.

    Input:
        1 x 28 x 28 grayscale image

    Output:
        10 classes
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.fc1 = nn.Linear(
            32 * 7 * 7,
            128
        )

        self.fc2 = nn.Linear(
            128,
            10
        )


    def forward(self, x):
        """
        Forward pass.

        Shape progression:

        Input:
        [batch,1,28,28]

        conv1:
        [batch,16,28,28]

        pool:
        [batch,16,14,14]

        conv2:
        [batch,32,14,14]

        pool:
        [batch,32,7,7]

        flatten:
        [batch,1568]

        fc layers:
        [batch,10]
        """

        x = self.conv1(x)
        x = F.relu(x)

        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.pool(x)

        x = torch.flatten(
            x,
            start_dim=1
        )

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x



def create_model():
    """
    Factory function.

    Every client should call this separately,
    creating its own independent model.
    """

    return SimpleCNN()