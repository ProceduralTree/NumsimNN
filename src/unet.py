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
        self.time_embedding = nn.Sequential(
            nn.Linear(embedding_dimension, in_ch * 2),
            nn.SiLU(),
            nn.Linear(in_ch * 2, in_ch * 2),
        )
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
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, t):
        skip = x
        gamma, mu = self.time_embedding(t).chunk(2, dim=1)
        x = x * (1 + gamma[:, :, None, None]) + mu[:, :, None, None]
        x = self.conv(x)
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        embedding_dimension,
        kernel_size=3,
        actv=nn.SiLU(),
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.time_embedding = nn.Sequential(
            nn.Linear(embedding_dimension, (in_ch + skip_ch) * 2),
            nn.SiLU(),
            nn.Linear((in_ch + skip_ch) * 2, (in_ch + skip_ch) * 2),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch + skip_ch,
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

    def forward(self, x, skip, t):
        gamma, mu = self.time_embedding(t).chunk(2, dim=1)
        x = self.up(x)  # upsample
        x = pt.cat([x, skip], 1)  # concatenate skip connection
        x = self.conv(
            x * (1 + gamma[:, :, None, None]) + mu[:, :, None, None]
        )  # refine
        return x


class UNET(nn.Module):

    def __init__(
        self,
        in_ch,
        out_ch,
        depth,
        kernel_size,
        input_shape,
        hidden_factor=10,
        embedding_dimension=10,
    ) -> None:
        super().__init__()
        self.depth = depth
        self.hidden_factor = hidden_factor
        self.shape = input_shape
        self.time_embedding = SinusoidalEmbedding(10, 10, embedding_dimension)
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.post_smoothing = nn.Sequential(
            nn.Conv2d(hidden_factor, hidden_factor, 3, padding=1, padding_mode="zeros"),
            nn.SiLU(),
            nn.Conv2d(hidden_factor, out_ch, 3, padding=1, padding_mode="zeros"),
        )
        self.encoder.append(EncoderBlock(in_ch, hidden_factor, embedding_dimension))
        self.decoder.append(
            DecoderBlock(hidden_factor, in_ch, hidden_factor, embedding_dimension)
        )
        for i in range(1, depth):
            input = hidden_factor * i
            output = hidden_factor * (i + 1)
            self.encoder.append(
                EncoderBlock(input, output, embedding_dimension, kernel_size)
            )
            self.decoder.append(
                DecoderBlock(output, input, input, embedding_dimension, kernel_size)
            )

        self.bottleneck = nn.Sequential(
            MultiheadAttention(hidden_factor * depth, hidden_factor * depth, 16),
            nn.SiLU(),
            MultiheadAttention(hidden_factor * depth, hidden_factor * depth, 16),
        )

    def init(
        self,
        init_fn=lambda x: (
            init.kaiming_normal_(x.weight)
            if isinstance(x, (nn.Conv2d, nn.ConvTranspose2d))
            else None
        ),
    ):
        self.apply(init_fn)

    def forward(self, x, t):
        skips = []
        t_embedd = self.time_embedding(t)
        for i in range(self.depth):
            x, skip = self.encoder[i](x, t_embedd)
            skips.append(skip)
        # x_linear: Tensor = x.flatten(start_dim=1)

        # x: Tensor = self.bottleneck(x)
        # x = x_linear.reshape(-1, self.hidden_factor, self.shape[0] // 2**self.depth, self.shape[1] // 2**self.depth,)

        for i in reversed(range(self.depth)):
            skip = skips.pop()
            x = self.decoder[i](x, skip, t_embedd)
        return self.post_smoothing(x)
