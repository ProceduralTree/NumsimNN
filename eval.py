import torch.nn as nn
import torch as pt
import matplotlib.pyplot as plt
import src.unet as unet
from importlib import reload
from torch.nn import init

plt.ioff()

reload(unet)
dev = pt.device("cpu")
if pt.xpu.is_available():
    print("Found Functional Intel GPU using dev=xpu")
    dev = pt.device("xpu")
if pt.cuda.is_available():
    print("Found Functional NVIDIA GPU using dev=cuda")
    dev = pt.device("cuda")

uNet = unet.UNET(3, 3, 3, 5,  hidden_factor=70 , input_shape=(32,32)).to(dev)
uNet.init()

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Subset
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32,32)),
    transforms.Normalize((0.5,), (0.5,))])
# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.CIFAR10('./data', train=True, transform=transform, download=True )
validation_set = torchvision.datasets.CIFAR10('./data', train=False, transform=transform, download=True)
train_loader = DataLoader(training_set , batch_size = 128 , shuffle=True)
val_loader = DataLoader(validation_set , batch_size = 128 , shuffle=False)

from src.train_diffusion import train , NoiseSchedule
import src.train_diffusion as tdf
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
reload(tdf)
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
N = NoiseSchedule(dev)
train(uNet , 1000 , train_loader , val_loader, writer, dev , N)

import matplotlib.pyplot as plt
import random
plt.imshow(validation_set[random.randint(0,len(validation_set))][0].permute(1,2,0) , cmap="Greys" , interpolation='bicubic')
