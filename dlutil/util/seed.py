import random
import numpy as np
import torch
import pytorch_lightning as pl


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def cudnn_deterministic():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed_pl(seed=42):
    pl.seed_everything(42)
