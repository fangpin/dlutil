import torch
import os


class ModelSaver:
    def __init__(
        self, model: torch.nn.Module, checkpoint_root_path=".", save_only_best=False
    ):
        self.model = model
        self.save_only_best = save_only_best
        self.checkpoint_root_path = checkpoint_root_path

    def save(self, epoch: int, is_best=False):
        if is_best:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_root_path, "model_best.pth"),
            )
        else:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoint_root_path, f"model_epoch_{epoch}.pth"),
            )
