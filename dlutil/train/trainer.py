from typing import Optional
from torch import device
import torch
from trainer_module import TrainerModule
from model_saver import ModelSaver
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np


class Trainer:
    def __init__(
        self,
        model: TrainerModule,
        epoch: int,
        train_loader: DataLoader,
        val_lader: Optional[DataLoader],
        test_loader: Optional[DataLoader],
        best_checker="",
        device,
    ) -> None:
        self.model = model
        self.epoch = epoch
        self.train_loader = train_loader
        self.val_loader = val_lader
        self.test_loader = test_loader
        self.device = device

    def train(self) -> None:
        optimizer, scheduler = self.model.config_optimizer()
        result = {}
        for epoch in tqdm(range(self.epoch)):
            self.model.train()

            losses = np.zeros(len(self.train_loader))
            for batch_idx, batch in enumerate(self.train_loader):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                optimizer.zero_grad()
                loss, result = self.model.training_step((inputs, targets), batch_idx)
                loss.backward()
                optimizer.step()
                losses[batch_idx](loss.item())

            scheduler.step()

            print(f"avg train loss is {losses.mean()} at epoch {epoch}")
            if result is not None and "acc" in result:
                print(f"avg train acc is {result['acc']} at epoch {epoch}")

            if self.val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    losses = np.zeros(len(self.val_loader))
                    for batch_idx, batch in enumerate(self.val_loader):
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)
                        loss, result = self.model.validation_step(
                            (inputs, targets), batch_idx
                        )
                        losses[batch_idx] = loss
                    print(f"avg val loss is {losses.mean()} at epoch {epoch}")
                    if result is not None and "acc" in result:
                        print(f"avg val acc is {result['acc']} at epoch {epoch}")

            if self.test_loader is not None:
                self.model.eval()
                test_losses = np.zeros(len(self.test_loader))
                with torch.no_grad():
                    for batch_idx, batch in enumerate(self.test_loader):
                        inputs, targets = batch
                        inputs, targets = inputs.to(device), targets.to(device)
                        loss, result = self.model.validation_step(
                            (inputs, targets), batch_idx
                        )
                        losses[batch_idx] = loss
                    print(f"avg test loss is {test_losses.mean()} at epoch {epoch}")
                    if result is not None and "acc" in result:
                        print(f"avg test acc is {result['acc']} at epoch {epoch}")
