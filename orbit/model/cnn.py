"""Compact 1D-CNN vibration fault classifier.

Architecture (~120k parameters): three stride-2 conv blocks over the raw
window followed by global average pooling. Small enough to train on CPU
in minutes, structured enough to benefit from batched GPU execution —
the profile the AMD benchmark targets.

The classifier sees raw vibration windows only. It is a *corroborating*
signal in the ORBIT workflow: deterministic evidence rules remain the
authority for diagnosis and action.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from orbit.model.dataset import CLASSES


class FaultCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), channels: Tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=4, stride=4),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)  # (batch, samples) -> (batch, 1, samples)
        return self.head(self.features(x))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
