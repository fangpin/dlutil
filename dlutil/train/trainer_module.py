import torch
import torch.nn as nn
from typing import Optional, Tuple

from types import MappingProxyType


class TrainerModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.epoch = 0
        self.iteration = 0
        self.best_metric = 0.0
        self.best_epoch = 0
        self.best_iteration = 0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.train_metric = 0.0
        self.val_metric = 0.0

    def config_optimizer(
        self,
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        raise NotImplementedError("config_optimizer")

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        raise NotImplementedError("trainint_step")

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        raise NotImplementedError("validation_step")

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        raise NotImplementedError("test_step")
