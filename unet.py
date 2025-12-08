#!/usr/bin/env ipython
import torch.nn as nn
import torch as pt


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
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
            actv,
        )

    def forward(self, x, skip):
        x = self.up(x)  # upsample
        x = pt.cat([x, skip], 1)  # concatenate skip connection
        x = self.conv(x)  # refine
        return x


class UNET(nn.Module):

    def __init__(self, in_ch, out_ch, depth, kernel_size, padding) -> None:
        super().__init__()
        self.depth = depth
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(depth):
            input = in_ch * 2**i
            output = out_ch * 2**i
            self.encoder.append(EncoderBlock(input, input * 2, kernel_size, padding))
            self.decoder.append(
                DecoderBlock(output * 2, input * 2, output, kernel_size, padding)
            )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch * 2**depth, in_ch * 2**depth, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 2**depth, out_ch * 2**depth, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x, skip = self.encoder[i](x)
            skips.append(skip)

        x = self.bottleneck(x)
        print(x.shape)

        for i in reversed(range(self.depth)):
            skip = skips.pop()
            x = self.decoder[i](x, skip)
        return x
