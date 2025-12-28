# Forward Diffusion
# \begin{align}
# \label{eq:1}
# x_t &= \lambda   x_0  + (1-\lambda  ) \mathcal{N}(0,I)
# \end{align}
# where \(\lambda \) is the noise schedule

# \begin{align}
# \label{eq:2}
# \lambda &= e^{-\int_{0}^t \beta (t) \, \mathrm{d}t }
# \end{align}
# with a cosine noise schedule
# \begin{align}
# \label{eq:3}
# \beta (t) &= \cos^2 \left( \frac{\pi }{2} \frac{t+\epsilon }{1 + \epsilon } \right)
# \end{align}

import torch as pt
from dataclasses import dataclass
class NoiseSchedule:

    def __init__(self,dev : pt.device ,  eps = 0.008 , T=100):
        # 1. Create time steps [0, T]
        # 2. Compute alpha_bar using cosine schedule
        self.T = T
        # self.t = pt.linspace(0 , self.T , self.T)
        # alpha_bar = pt.cos((self.t / self.T + eps) / (1 + eps) * pt.pi / 2) ** 2
        # alpha_bar = alpha_bar / alpha_bar[0]  # normalize so alpha_bar[0] = 1

        # beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        # beta= pt.clip(beta, 0, 0.999)

        # self.beta = beta.to(dev)
        self.beta = pt.linspace(1e-4 , 0.02 , self.T)
        self.alpha = (1-self.beta).to(dev)
        self.alpha_bar = pt.cumprod(self.alpha, dim=0).to(dev)



# #+RESULTS:



import torch as pt
import torch.nn as nn
from torch import Tensor
    
def diffusion_step(x : Tensor , model : nn.Module,dev, N):
    """
    data : minibatch with label and input data:
        data[0].shape = (Batchsize , channels , xsize , ysize )
    model: trainings model for example uNet
    """
    index = pt.randint(0, len(N.beta) , (x.size(0),) , device=dev)
    noise = pt.randn_like(x, device=dev)
    noisy_input = pt.sqrt(1-N.alpha_bar[index]).view(-1,1,1,1) *  noise + pt.sqrt(N.alpha_bar[index]).view(-1,1,1,1) * x
    
    
    output = model(noisy_input) 
    return output , noise



# #+RESULTS: alphabars.png
# [[file:alphabars.png]]


import torch as pt
import torch.nn as nn
from torch import Tensor
def sample_diffusion(T:int,model: nn.Module , shape:tuple , dev:pt.device , N):
   x = pt.randn(shape , device=dev)
   for t in reversed(range(0,T)):
      with pt.no_grad():
         predicted_noise = model(x)
      noise = pt.randn(shape , device=dev)
      sigma = pt.sqrt(N.beta[t])
      weighted_noise = ((1-N.alpha[t].view(-1,1,1,1))/pt.sqrt(1.-N.alpha_bar[t])) * predicted_noise
      x = (1/pt.sqrt(N.alpha[t]).view(-1,1,1,1)) * (x - weighted_noise)
      if t>0:
         x += sigma * noise

   return x

# Diffusion training structure


from tqdm import trange , tqdm
import torch as pt
from torch.utils.data import DataLoader
from torch.nn import Module , MSELoss
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer, AdamW
from pathlib import Path

def train(
    model: Module,
    epochs: int,
    train_data: DataLoader,
    validation_data: DataLoader,
    writer: SummaryWriter,
    dev,
    N
):
    optimizer = AdamW(model.parameters(), lr=1e-4)
    if Path("models/checkpoint").exists():
        model.load_state_dict(pt.load("models/checkpoint"))
    if Path("optim/checkpoint").exists():
        optimizer.load_state_dict(pt.load("optim/checkpoint"))
    loss_fn = MSELoss()
    min_loss = 10000.
    for e in trange(epochs):
        
        model.train(True)
        running_loss :float = 0.0
        for i,data in tqdm(enumerate(train_data) , total=len(train_data)):
            x , _ = data
            x=x.to(dev)
            optimizer.zero_grad()
            output , noise = diffusion_step(x , model , dev , N)
            loss = loss_fn(noise , output)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i% 100 == 99:
                last_loss = running_loss / 100  # loss per batch
                tb_x = e * len(data) + i + 1
                writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0


        validation_loss = 0.
        for i , data in enumerate(validation_data):
            with pt.no_grad():
                input, label = data
                input=input.to(dev)
                output , noise = diffusion_step(input , model, dev , N)
                loss = loss_fn(noise , output)
                validation_loss += loss.item()
        validation_loss /= len(validation_data)
        if validation_loss < min_loss :
            pt.save(model.state_dict() , f"models/checkpoint")
            pt.save(optimizer.state_dict() , f"optim/checkpoint")
            min_loss= validation_loss
        writer.add_scalar("Loss/val", validation_loss, e)
