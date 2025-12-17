import torch.nn as nn
import torch as pt
import matplotlib.pyplot as plt
import unet
from importlib import reload
from torch.nn import init

plt.ioff()

reload(unet)
dev = "cpu"
if pt.xpu.is_available():
    print("Found Functional Intel GPU using dev=xpu")
    dev = "xpu"

uNet = unet.UNET(1, 1, 4, 3, 1).to(dev)
uNet.init()
x = pt.rand(1, 1, 128, 128).to(dev)
x

fig , ax = plt.subplots()
ax.imshow(x[0,0,:,:].to("cpu").detach().squeeze())
ax.set_xlabel("x")
ax.set_ylabel("y")
fig

plt.close('all')
input, output = validation_set[pt.randint(0, 1000, ())]
y = uNet(input.unsqueeze(1).to(dev))
fig , ax = plt.subplots(1,3 ,figsize=(9,3))
ax[0].imshow(input[0,:,:].to("cpu").detach().squeeze() , cmap="Greys")
ax[0].set_title("Original")
ax[1].imshow(y[0,0,:,:].to("cpu").detach().squeeze() , cmap="Greys")
ax[1].set_title("Prediction")
ax[2].imshow((input.to(dev)-y)[0,0,:,:].to("cpu").detach().squeeze() , cmap="gray")
ax[2].set_title("Difference")
for i in range(3):
     ax[i].set_xlabel("x")
     ax[i].set_ylabel("y")
fig

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((64,64)),
    transforms.Normalize((0.5,), (0.5,))])
# Create datasets for training & validation, download if necessary
training_set = torchvision.datasets.FashionMNIST('./data', train=True, transform=transform, download=True )
validation_set = torchvision.datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
train_loader = DataLoader(training_set , batch_size = 128 , shuffle=True)
val_loader = DataLoader(validation_set , batch_size = 128 , shuffle=False)

import training
from training import train_step , train
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

reload(training)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
train(uNet , 10 , train_loader , val_loader , writer, dev)
