#!/usr/bin/env ipython
import torch.nn as nn
from torch.nn import Conv2d, ConvTranspose2d, init
import torch as pt


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.pool(x)
        return x, skip


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_ch,
        skip_ch,
        out_ch,
        kernel_size=3,
        padding=1,
        actv=nn.ReLU(inplace=True),
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size, padding=padding),
            actv,
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
        )

    def forward(self, x, skip):
        x = self.up(x)  # upsample
        x = pt.cat([x, skip], 1)  # concatenate skip connection
        x = self.conv(x)  # refine
        return x


class UNET(nn.Module):

    def __init__(
        self, in_ch, out_ch, depth, kernel_size, padding, hidden_factor=10
    ) -> None:
        super().__init__()
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.encoder.append(EncoderBlock(in_ch, hidden_factor, kernel_size, padding))
        self.decoder.append(
            DecoderBlock(
                hidden_factor,
                hidden_factor,
                out_ch,
                kernel_size,
                padding,
            )
        )
        for i in range(1, depth):
            input = hidden_factor
            output = hidden_factor
            self.encoder.append(EncoderBlock(input, input, kernel_size, padding))
            self.decoder.append(
                DecoderBlock(output, input, output, kernel_size, padding)
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(
                hidden_factor,
                hidden_factor * 10,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_factor * 10,
                hidden_factor,
                kernel_size=3,
                padding=1,
            ),
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

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x, skip = self.encoder[i](x)
            skips.append(skip)

        x = self.bottleneck(x)

        for i in reversed(range(self.depth)):
            skip = skips.pop()
            x = self.decoder[i](x, skip)
        return x
