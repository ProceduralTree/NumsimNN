#!/usr/bin/env ipython
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, init
import torch as pt
from .embedding import SinusoidalEmbedding
from .attention import MultiheadAttention
from torch import Tensor


class CNN(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        layers = [nn.Conv2d(1, 16, 7, padding="same", stride=1), nn.ReLU()]
        hiddenLayers = [
            nn.Conv2d(16, 16, 7, padding="same", stride=1),
            nn.ReLU(),
        ]
        hiddenLayers *= 5
        layers += hiddenLayers
        layers.append(MultiheadAttention(16, 16, 8))
        layers.append(nn.Conv2d(16, 2, 7, padding="same", stride=1))
        self.conv = nn.Sequential(*layers)

    def forward(
        self,
        x: Tensor,
    ):
        return self.conv(x)
