import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_epoch):
        self.warmup = warmup
        self.max_epoch = max_epoch
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_epoch))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


if __name__ == "__main__":
    p = nn.Parameter(torch.empty(4, 4))
    optimizer = optim.Adam([p], lr=1e-3)
    lr_scheduler = CosineWarmupScheduler(
        optimizer=optimizer, warmup=100, max_epoch=2000
    )

    epochs = list(range(2000))
    print([lr_scheduler.get_lr_factor((e)) for e in epochs])
