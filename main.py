from src.cnn import CNN
import torch.nn as nn
import torch as pt
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import src.unet as unet
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from src.training import train  # ZUG
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

pt.serialization.add_safe_globals([pt.utils.data.dataset.Subset])
pt.serialization.add_safe_globals([TensorDataset])
dev = pt.device("cpu")
if pt.xpu.is_available():
    print("Found Functional Intel GPU using dev=xpu")
    dev = pt.device("xpu")
if pt.cuda.is_available():
    print("Found Functional NVIDIA GPU using dev=cuda")
    dev = pt.device("cuda")


training_set = pt.load("./data/train_data.pt")
validation_set = pt.load("./data/validation_data.pt")
train_loader = DataLoader(training_set, batch_size=16, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=16, shuffle=False)


cnn = CNN().to(dev)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter("runs/numsim_trainer_{}".format(timestamp))
train(cnn, 4000, train_loader, val_loader, writer, dev)
