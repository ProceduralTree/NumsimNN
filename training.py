#!/usr/bin/env ipython
#
import torch as pt
from torch.optim import Optimizer, AdamW
from torch.nn import Module
from torch import Tensor
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from _collections_abc import Callable
from torch.utils.data import DataLoader


def train(
    model: Module,
    epochs: int,
    train_data: DataLoader,
    validation_data: DataLoader,
    writer: SummaryWriter,
    dev,
):
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = MSELoss()
    for e in range(epochs):
        model.train(True)
        train_step(model, optimizer, loss_fn, train_data, e, writer, dev)
        # validation_batch(model, validation_data, writer, dev, loss_fn, e)


def train_step(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable,
    data,
    epoch,
    writer: SummaryWriter,
    dev,
):
    running_loss = 0.0
    last_loss = 0.0
    for i, data in enumerate(data):
        input, label = data
        input = input.to(dev)
        optimizer.zero_grad()
        output = model(input)
        loss: Tensor = loss_fn(input, output)
        loss.backward()
        optimizer.step()
        # Gather data and report
        running_loss += loss.item()
        if i % 100 == 99:
            last_loss = running_loss / 1000  # loss per batch
            print("  batch {} loss: {}".format(i + 1, last_loss))
            tb_x = epoch * len(data) + i + 1
            writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0


def validation_batch(
    model: Module,
    validation_data: DataLoader,
    writer: SummaryWriter,
    dev,
    loss_fn: Callable,
    epoch: int,
):
    running_vloss = 0.0
    avg_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with pt.no_grad():
        for i, vdata in enumerate(validation_data):
            vinputs, vlabels = vdata
            voutputs = model(vinputs.to(dev))
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars(
        "Training vs. Validation Loss", {"Validation": avg_vloss}, epoch + 1
    )
    writer.flush()
