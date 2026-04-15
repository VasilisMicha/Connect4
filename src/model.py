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
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)       

        self.dropout = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(128 * rows * columns, 256)
        self.fc2 = nn.Linear(256, actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
