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
):

    # ensure storage paths exist
    Path("models").mkdir(parents=True , exist_ok=True)
    Path("optim").mkdir(parents=True , exist_ok=True)
    
    optimizer = AdamW(model.parameters(), lr=1e-4)
    if Path("models/checkpoint").exists():
        print(f"Found Model Checkpoint. Loading Checkpoint")
        model.load_state_dict(pt.load("models/checkpoint"))
    if Path("optim/checkpoint").exists():
        print(f"Found Optimizer Checkpoint. Loading Checkpoint")
        optimizer.load_state_dict(pt.load("optim/checkpoint"))
    loss_fn = MSELoss()
    min_loss = 10000.
    for e in trange(epochs):
        
        model.train(True)
        running_loss :float = 0.0
        for i,data in tqdm(enumerate(train_data) , total=len(train_data)):
            x , label = data
            x=x.to(dev)
            optimizer.zero_grad()
            prediction = model(x)

            loss = loss_fn(prediction , label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i% 100 == 99:
                last_loss = running_loss / (i+1)  # loss per batch
                tb_x = e * len(data) + i + 1
                running_loss = 0.0

        writer.add_scalar("Loss/train", last_loss, e)

        validation_loss = 0.
        for i , data in enumerate(validation_data):
            with pt.no_grad():
                input, label = data
                input=input.to(dev)
                validation_loss += loss.item()
        validation_loss /= len(validation_data)
        if validation_loss < min_loss :
            pt.save(model.state_dict() , f"models/checkpoint")
            pt.save(optimizer.state_dict() , f"optim/checkpoint")
            min_loss= validation_loss
        writer.add_scalar("Loss/val", validation_loss, e)
        if e%10 ==0:
            input , output = next(iter(validation_data))
            writer.add_image("input" , input[0],e )
            writer.add_image("output" , output[0],e )
