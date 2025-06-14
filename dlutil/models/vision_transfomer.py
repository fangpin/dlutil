from os import devnull
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    import os
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    import torch.utils.data as data

    CHECK_POINT_PATH = "../saved_models"
    DATASET_PATH = "../data"
    pl.seed_everything(42)
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
    pl.seed_everything(42)
    train_set, _ = torch.utils.data.random_split(train_dataset, [45000, 5000])
    pl.seed_everything(42)
    _, val_set = torch.utils.data.random_split(val_dataset, [45000, 5000])

    # Loading the test set
    test_set = CIFAR10(
        root=DATASET_PATH, train=False, transform=test_transform, download=True
    )

    # We define a set of data loaders that we can use for various purposes later.
    train_loader = data.DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=128,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )
    test_loader = data.DataLoader(
        test_set, batch_size=128, shuffle=False, drop_last=False, num_workers=4
    )

    class ViT(pl.LightningModule):
        def __init__(self, kwargs, lr: float):
            super().__init__()
            self.lr = lr
            self.model = VisionTransformer(**kwargs)
            # self.example_input_array = next(iter(train_loader))[0]

        def forward(self, x) -> torch.Tensor:
            return self.model.forward(x)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[100, 150], gamma=0.1
            )
            return [optimizer], [lr_scheduler]

        def _calculate_loss(self, batch, mode: str = "train"):
            x, y = batch
            preds = self.forward(x)
            loss = F.cross_entropy(preds, y)
            acc = (preds.argmax(dim=-1) == y).float().mean()
            self.log(f"{mode}_loss", loss)
            self.log(f"{mode}_acc", acc)
            return loss

        def training_step(self, batch, batch_idx):
            return self._calculate_loss(batch, mode="train")

        def validation_step(self, batch, batch_idx):
            self._calculate_loss(batch, mode="val")

        def test_step(self, batch, batch_idx):
            self._calculate_loss(batch, mode="test")

    def train_model(**kwargs):
        trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECK_POINT_PATH, "ViT"),
            accelerator="gpu" if str(device).startswith("cuda") else "cpu",
            devices=1,
            max_epochs=180,
            callbacks=[
                ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                LearningRateMonitor("epoch"),
            ],
        )

        pretained_filename = os.path.join(CHECK_POINT_PATH, "ViT.ckpt")
        if os.path.isfile(pretained_filename):
            print(f"found pretrained model at {pretained_filename}, loading...")
            model = ViT.load_from_checkpoint(pretained_filename)
        else:
            pl.seed_everything(42)
            model = ViT(**kwargs)
            trainer.fit(model, train_loader, val_loader)
            model = ViT.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path
            )

        val_result = trainer.test(model, val_loader, verbose=False)

        test_result = trainer.test(model, test_loader, verbose=False)
        result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

        return model, result

    model, results = train_model(
        kwargs={
            "embed_dim": 256,
            "forward_dim": 512,
            "num_heads": 8,
            "num_attention_layers": 6,
            "patch_size": 4,
            "num_channel": 3,
            "num_patches": 64,
            "num_class": 10,
            "dropout": 0.2,
        },
        lr=3e-4,
    )
    print("ViT results", results)
