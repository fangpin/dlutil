import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dlutil.util as util


def img_to_patches(img: torch.Tensor, patch_size: int):
    batch, c, h, w = img.shape
    img = img.reshape(
        batch, c, h // patch_size, patch_size, w // patch_size, patch_size
    )
    img = img.permute(
        0, 2, 4, 1, 3, 5
    )  # [batch, h/patch_size, w/patch_size, c, patch_size, patch_size]

    img = img.flatten(
        3, 5
    )  # [batch, h/patch_size, w/patch_size, c * patch_size * patch_size]
    img = img.flatten(
        1, 2
    )  # [batch_size, h/patch_size * w/patch_size, c * patch_size * patch_size]
    return img


class AttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        forward_dim: int,
        num_heads: int,
        dropout=0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.atten_layer = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.liner = nn.Sequential(
            nn.Linear(embed_dim, forward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(forward_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        input_x = self.norm1(x)
        x = x + self.atten_layer(input_x, input_x, input_x)[0]
        x = x + self.liner(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        num_channel: int,
        num_patches: int,
        num_attention_layers: int,
        embed_dim: int,
        num_heads: int,
        forward_dim: int,
        num_class=10,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = patch_size * patch_size * num_channel
        self.input_layer = nn.Linear(self.input_dim, embed_dim)
        self.transformer = nn.Sequential(
            *[
                AttentionBlock(embed_dim, forward_dim, num_heads, dropout)
                for _ in range(num_attention_layers)
            ]
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_class),
        )
        self.dropout = nn.Dropout(dropout)

        self.position_encoding = nn.Parameter(
            torch.randn(
                1,
                num_patches + 1,
                embed_dim,
            )
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x: torch.Tensor):
        x = img_to_patches(x, self.patch_size)
        batch, seq_num, input_dim = x.shape
        assert input_dim == self.input_dim

        embeding = self.input_layer(x)
        cls_token = self.cls_token.repeat((batch, 1, 1))
        embeding = torch.cat([cls_token, embeding], dim=1)
        embeding = embeding + self.position_encoding[:, : seq_num + 1]
        embeding = self.dropout(embeding)
        embeding = embeding.permute(
            1, 0, 2
        )  # nn.MultiheadAttention use [seq, batch, feature] format

        cls = self.transformer(embeding)[0]
        out = self.mlp(cls)
        return out


if __name__ == "__main__":
    import os
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    import torch.utils.data as data
    from tqdm.auto import tqdm
    import numpy as np
    from torch.utils.tensorboard.writer import SummaryWriter

    CHECK_POINT_PATH = "../saved_models"
    LOG_PATH = "../logs"
    DATASET_PATH = "../data"
    BATCH_SIZE = 128
    util.seed_everything(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    print("Device:", device)

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [0.49139968, 0.48215841, 0.44653091],
                [0.24703223, 0.24348513, 0.26158784],
            ),
        ]
    )
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.49139968, 0.48215841, 0.44653091],
                [0.24703223, 0.24348513, 0.26158784],
            ),
        ]
    )
    # Loading the training dataset. We need to split it into a training and validation part
    # We need to do a little trick because the validation set should not use the augmentation.
    train_dataset = CIFAR10(
        root=DATASET_PATH, train=True, transform=train_transform, download=True
    )
    val_dataset = CIFAR10(
        root=DATASET_PATH, train=True, transform=test_transform, download=True
    )
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(
        root=DATASET_PATH, train=False, transform=test_transform, download=True
    )

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = data.DataLoader(
        test_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4
    )

    model = VisionTransformer(
        embed_dim=256,
        forward_dim=512,
        num_heads=8,
        num_attention_layers=6,
        patch_size=4,
        num_channel=3,
        num_patches=64,
        num_class=10,
        dropout=0.2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 150], gamma=0.1
    )

    def train_model(
        model: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        max_epoch: int,
    ):
        pretained_filename = os.path.join(CHECK_POINT_PATH, "ViT.ckpt")
        log_dir = os.path.join(LOG_PATH, "ViT")
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        if os.path.isfile(pretained_filename):
            print(f"found pretrained model at {pretained_filename}, loading...")
            model.load_state_dict(torch.load(pretained_filename))
        else:
            best_acc = 0.0
            global_step = 0
            for epoch in range(max_epoch):
                model.train()
                all_loss = np.zeros(len(train_loader))
                for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
                    optimizer.zero_grad()
                    inputs, targets = inputs.to(device), targets.to(device)
                    preds = model.forward(inputs)
                    loss = F.cross_entropy(preds, targets)
                    loss.backward()
                    optimizer.step()
                    all_loss[batch_idx] = loss.item()
                    writer.add_scalar("train_loss", loss.item(), global_step)
                    cur_lr = optimizer.param_groups[0]["lr"]
                    writer.add_scalar("lr", cur_lr, global_step)
                    global_step += 1
                scheduler.step()
                print(f"train avg loss {all_loss.mean()} at epoch {epoch}")

                model.eval()
                num_sampels, num_correct = 0, 0
                val_acc = 0.0
                with torch.no_grad():
                    all_loss = np.zeros(len(val_loader))
                    for batch_idx, (inputs, targets) in enumerate(val_loader):
                        inputs, targets = inputs.to(device), targets.to(device)
                        preds = model.forward(inputs)
                        loss = F.cross_entropy(preds, targets)
                        all_loss[batch_idx] = loss.item()
                        num_sampels += targets.size(0)
                        num_correct += (torch.argmax(preds, dim=-1) == targets).sum()
                    val_acc = num_correct / num_sampels
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(
                            model.state_dict(),
                            os.path.join(CHECK_POINT_PATH, "ViT.ckpt"),
                        )
                    print(f"val avg loss {all_loss.mean()} at epoch {epoch}")
                    print(f"val correct rate {val_acc} at epoch {epoch}")
                    writer.add_scalar("val_acc", val_acc, epoch)

            model.eval()
            with torch.no_grad():
                num_sampels, num_correct = 0, 0
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    preds = model.forward(inputs)
                    loss = F.cross_entropy(preds, targets)
                    num_sampels += targets.size(0)
                    num_correct += (torch.argmax(preds, dim=-1) == targets).sum()
                print(f"final test correct rate {num_correct / num_sampels}")

    model = VisionTransformer(
        patch_size=4,
        num_channel=3,
        num_patches=64,
        num_attention_layers=6,
        embed_dim=256,
        num_heads=8,
        forward_dim=512,
        num_class=10,
        dropout=0.2,
    )
    model = model.to(device)
    print(f"number of parameters: {util.num_params(model)}")

    train_model(
        model,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        max_epoch=180,
    )
