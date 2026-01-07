from src.cnn import CNN
import torch.nn as nn
import torch as pt
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import src.unet as unet
from torch.nn import init

dev = pt.device("cpu")
if pt.xpu.is_available():
    print("Found Functional Intel GPU using dev=xpu")
    dev = pt.device("xpu")
if pt.cuda.is_available():
    print("Found Functional NVIDIA GPU using dev=cuda")
    dev = pt.device("cuda")

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

# Create datasets for training & validation, download if necessary
data = pt.load("data/training_data.pt")
input = [data["training_in"], data["training_out"]]
data_set = TensorDataset(input[0], input[1])
split_size_training = int(len(data_set) * 0.8)
split_size_val = int(len(data_set) * 0.1)
split_size_test = len(data_set) - split_size_training - split_size_val
training_set, validation_set, test_set = random_split(
    data_set, [split_size_training, split_size_val, split_size_test]
)
train_loader = DataLoader(training_set, batch_size=16, shuffle=True)
val_loader = DataLoader(validation_set, batch_size=16, shuffle=False)

from src.training import train  # ZUG
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

cnn = CNN().to(dev)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter("runs/numsim_trainer_{}".format(timestamp))
train(cnn, 4000, train_loader, val_loader, writer, dev)
