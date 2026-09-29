"""Frozen visual features and frame vs temporal tool-presence heads."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class FeatureEncoder(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        net = mobilenet_v3_small(weights=weights)
        self.features = net.features
        self.pool = net.avgpool
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode: bool = True):
        # Features stay frozen in eval mode so batch-norm uses running stats.
        return super().train(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.pool(self.features(x)).flatten(1)


class FramePresence(nn.Module):
    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TemporalPresence(nn.Module):
    def __init__(self, dim: int, hidden: int, num_classes: int):
        super().__init__()
        self.gru = nn.GRU(dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)
        return self.fc(out[:, -1])
