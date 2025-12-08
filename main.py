import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import unet
from importlib import reload

reload(unet)


dev = "cpu"
if torch.xpu.is_available():
    print("Found Functional Intel GPU using dev=xpu")
    dev = "xpu"


def main():
    print("Hello from numsimnn!")


if __name__ == "__main__":
    enc = unet.EncoderBlock(2, 4, 3, 1).to(dev)
    dec = unet.DecoderBlock(4, 4, 2, 3, 1).to(dev)
    uNet = unet.UNET(2, 1, 3, 3, 1).to(dev)
    x = torch.rand(1, 2, 128, 128).to(dev)
    x_p, skip = enc(x)
    with torch.no_grad():
        plt.imshow(x_p.to("cpu").detach().squeeze())
        plt.imshow(skip.to("cpu").detach().squeeze())
    main()
