"""Lightweight CNN classifier for smoke / fog frames.

Report 4.2 calls for "a lightweight CNN classifier" for smoke/fog. This is a
~120k-parameter network suitable for edge devices (Jetson Nano class,
report 3.4). Train it with scripts/train_smoke.py on a folder dataset
(e.g. frames extracted from RESIDE / RTTS or any smoke-vs-clear image set).

Input : 3x128x128 float tensor in [0, 1]
Output: single logit (sigmoid > 0.5 => smoke/fog present)
"""

import torch
import torch.nn as nn


class SmokeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3, 16),    # 128 -> 64
            block(16, 32),   # 64 -> 32
            block(32, 64),   # 32 -> 16
            block(64, 64),   # 16 -> 8
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)
