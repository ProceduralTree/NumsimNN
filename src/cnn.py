#!/usr/bin/env ipython
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, init
import torch as pt
from .embedding import SinusoidalEmbedding
from .attention import MultiheadAttention
from torch import Tensor


class EncoderBlock(nn.Module):
    def __init__(
        self, in_ch, out_ch, embedding_dimension, kernel_size=3, actv=nn.SiLU()
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="replicate",
            ),
            actv,
            nn.Conv2d(
                out_ch,
                out_ch,
                kernel_size,
                padding=kernel_size // 2,
                padding_mode="replicate",
            ),
        )

    def forward(self, x, t):
        x = self.conv(x)
        return x


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
