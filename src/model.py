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
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout(p=0.2) # reduced from 0.3
        self.fc1 = nn.Linear(128 * rows * columns, 128) # wider
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
