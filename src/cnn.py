#!/usr/bin/env ipython
import torch.nn as nn
from torch.nn import Conv2d
import torch as pt
from .attention import MultiheadAttention, ConvBlockAttention
from torch import Tensor


class ResLayer(nn.Module):

    def __init__(self, input: int, out: int, hidden: int, kernel) -> None:
        super().__init__()
        self.res = nn.Identity() if input == out else nn.Conv2d(input, out, 1)
        self.relu = nn.ReLU()
        hiddenLayers = [
            nn.Conv2d(hidden, out, kernel, padding="same", stride=1),
            nn.ReLU(),
        ]
        hiddenLayers *= 3
        hiddenLayers.append(ConvBlockAttention(16))
        self.conv = nn.Sequential(*hiddenLayers)

    def forward(self, x: Tensor):
        return self.conv(x) + self.res(x)


class CNN(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        layers = [nn.Conv2d(1, 16, 7, padding="same", stride=1), nn.ReLU()]
        # hiddenLayers = [
        # nn.Conv2d(16, 16, 7, padding="same", stride=1),
        # nn.ReLU(),
        ## ConvBlockAttention(16),
        # ]
        # hiddenLayers *= 5
        layers += [ResLayer(16, 16, 16, 3) for i in range(15)]
        # layers += hiddenLayers
        layers.append(MultiheadAttention(16, 16, 8))
        layers.append(nn.Conv2d(16, 2, 7, padding="same", stride=1))
        self.conv = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
    ):
        return self.conv(x)
