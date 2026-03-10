import torch
from torch import nn
import torch.nn.functional as F
from typing import Final

rows: Final = 6
columns: Final = 7
channels: Final = 2

class DQN(nn.Module):

    def __init__(self, actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=64, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(64 * rows * columns, actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x)
        x = self.fc1(x)
        return x

