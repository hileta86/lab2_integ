"""Compact CNN architecture for keyword spotting on Mel-spectrograms."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioCNN(nn.Module):
    """Convolutional neural network for 4-class keyword classification."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.reshape(x.size(0), 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
