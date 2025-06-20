from flax.core import apply
import jax
import jax.numpy as jnp
import flax.linen as nn
from torch.utils.tensorboard.writer import SummaryWriter
import optax
from flax.training import train_state, checkpoints
from collections import defaultdict
import os
import numpy as np

CHECK_POINT_PATH = "../saved_models"
LOG_PATH = "../logs"
DATASET_PATH = "../data"
BATCH_SIZE = 128


def img_to_patches(x, patch_size):
    """
    Inputs:
        x - torch.Tensor representing the image of shape [B, H, W, C]
        patch_size - Number of pixels per dimension of the patches (integer)
                           as a feature vector instead of a image grid.
    """
    B, H, W, C = x.shape
    # x = x.permute(0, 2, 3, 1)
    # print(f"image batch data shape: {x.shape}")
    x = x.reshape(B, H // patch_size, patch_size, W // patch_size, patch_size, C)
    x = x.transpose(0, 1, 3, 2, 4, 5)  # [B, H', W', p_H, p_W, C]
    x = x.reshape(B, -1, *x.shape[3:])  # [B, H'*W', p_H, p_W, C]
    x = x.reshape(B, x.shape[1], -1)  # [B, H'*W', p_H*p_W*C]
    return x


class AttentionBlock(nn.Module):
    embed_dim: int
    hidden_dim: int
    num_heads: int
    dropout_rate: float = 0.0

    def setup(self):
        self.attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)
        self.linear = [
            nn.Dense(self.hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout_rate),
            nn.Dense(self.embed_dim),
        ]
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, train=True):
        x = self.norm1(x)
        attn_out = self.attn(x, x, x)
        x = x + self.dropout(attn_out, deterministic=not train)

        linear_out = self.norm2(x)
        for layer in self.linear:
            linear_out = (
                layer(linear_out)
                if not isinstance(layer, nn.Dropout)
                else layer(linear_out, deterministic=not train)
            )
        x = x + self.dropout(linear_out, deterministic=not train)
        return x


class VisionTransformer(nn.Module):
    embed_dim: int
    hidden_dim: int
    num_heads: int
    num_channels: int
    num_layers: int
    num_classes: int
    patch_size: int
    num_patches: int
    dropout_rate: float

    def setup(self):
        self.input_layer = nn.Dense(self.embed_dim)
        self.transformer = [
            AttentionBlock(
                self.embed_dim, self.hidden_dim, self.num_heads, self.dropout_rate
            )
            for _ in range(self.num_layers)
        ]
        self.mlp = nn.Sequential([nn.LayerNorm(), nn.Dense(self.num_classes)])
        self.dropout = nn.Dropout(self.dropout_rate)
        self.cls_token = self.param(
            "cls_token", nn.initializers.normal(stddev=1.0), (1, 1, self.embed_dim)
        )
        self.pos_embedding = self.param(
            "pos_embedding",
            nn.initializers.normal(stddev=1.0),
            (1, self.num_patches + 1, self.embed_dim),
        )

    def __call__(self, x, train=True):
        x = img_to_patches(x, self.patch_size)
        batch, seq, _ = x.shape
        x = self.input_layer(x)

        cls_token = self.cls_token.repeat(batch, axis=0)
        input_tokens = jnp.concatenate([cls_token, x], axis=1)
        embeding_out = input_tokens + self.pos_embedding[:, : seq + 1]
        embeding_out = self.dropout(embeding_out, deterministic=not train)
        for block in self.transformer:
            embeding_out = block(embeding_out, train)

        cls = embeding_out[:, 0]
        out = self.mlp(cls)
        return out


class ViTTrainer:
    seed: int = 42
    weight_decay: float = 0.01
    embed_dim: int
    hidden_dim: int
    num_heads: int
    num_channels: int
    num_layers: int
    num_classes: int
    patch_size: int
    num_patches: int

    def __init__(
        self,
        exmp_imgs,
        embed_dim,
        hidden_dim,
        num_heads,
        num_channels,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        lr=3e-4,
        dropout_rate=0.2,
    ):
        self.model = VisionTransformer(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_classes=num_classes,
            num_channels=num_channels,
            num_layers=num_layers,
            patch_size=patch_size,
            num_patches=num_patches,
            dropout_rate=dropout_rate,
        )
        self.rng = jax.random.PRNGKey(self.seed)
        self.log_dir = os.path.join(CHECK_POINT_PATH, "vit_jax/")
        self.logger = SummaryWriter(self.log_dir)
        self.lr = lr
        self.create_funcs()
        self.init_model(exmp_imgs)

    def create_funcs(self):
        def calculate_loss(params, rng, batch, train=True):
            imgs, labels = batch
            rng, dropout_rng = jax.random.split(rng)
            logits = self.model.apply(
                {"params": params}, imgs, train=train, rngs={"dropout": dropout_rng}
            )
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            ).mean()
            acc = (logits.argmax(axis=-1) == labels).mean()
            return loss, (acc, rng)

        def train_step(state, rng, batch):
            loss_fn = lambda params: calculate_loss(params, rng, batch, train=True)
            (loss, (acc, rng)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                state.params
            )
            state = state.apply_gradients(grads=grads)
            return state, rng, loss, acc

        def eval_step(state, rng, batch):
            _, (acc, rng) = calculate_loss(state.params, rng, batch, train=False)
            return rng, acc

        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)

    def init_model(self, exmp_imgs):
        self.rng, init_rng, dropout_rng = jax.random.split(self.rng, 3)
        self.init_params = self.model.init(
            {"params": init_rng, "dropout": dropout_rng}, exmp_imgs, train=True
        )["params"]
        self.state = None

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        lr_scheduler = optax.piecewise_constant_schedule(
            init_value=self.lr,
            boundaries_and_scales={
                int(num_steps_per_epoch * num_epochs * 0.6): 0.1,
                int(num_steps_per_epoch * num_epochs * 0.85): 0.1,
            },
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr_scheduler, weight_decay=self.weight_decay),
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=self.init_params if self.state is None else self.state.params,
            tx=optimizer,
        )

    def train_epoch(self, epoch):
        metrics = defaultdict(list)
        for batch in tqdm(train_loader, desc="training", leave=False):
            self.state, self.rng, loss, acc = self.train_step(
                self.state, self.rng, batch
            )
            metrics["loss"].append(loss)
            metrics["acc"].append(acc)

        avg_loss = np.stack(jax.device_get(metrics["loss"])).mean()
        self.logger.add_scalar("train/" + "loss", avg_loss, global_step=epoch)
        avg_acc = np.stack(jax.device_get(metrics["acc"])).mean()
        self.logger.add_scalar("train/" + "acc", avg_acc, global_step=epoch)
        return avg_loss, avg_acc

    def eval_model(self, data_loader):
        correct_class, count = 0, 0
        for batch in data_loader:
            self.rng, acc = self.eval_step(self.state, self.rng, batch)
            correct_class += acc * batch[0].shape[0]
            count += batch[0].shape[0]
        eval_acc = (correct_class / count).item()
        return eval_acc

    def save_model(self, step=0):
        checkpoints.save_checkpoint(
            ckpt_dir=self.log_dir, target=self.state, step=step, overwrite=True
        )

    def load_model(self, pretrained=False):
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            params = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join(CHECK_POINT_PATH, "ViT.ckpt"), target=None
            )
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=params,
            tx=self.state.tx if self.state else optax.adamw(self.lr),
        )

    def checkpoint_exists(self):
        return os.path.isfile(os.path.join(CHECK_POINT_PATH, "ViT,ckpt"))

    def train_model(self, train_loader, val_loader, test_loader, num_epochs=180):
        self.init_optimizer(num_epochs, len(train_loader))
        best_eval = 0.0
        for epoch_idx in tqdm(range(num_epochs)):
            loss, _ = self.train_epoch(epoch=epoch_idx)
            acc = self.eval_model(val_loader)
            print(f"avg loss {loss}, acc {acc}, at epoch {epoch_idx}")
        acc = self.eval_model(test_loader)
        print(f"final acc {acc}")


DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])


def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255.0 - DATA_MEANS) / DATA_STD
    return img


# We need to stack the batch elements
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


if __name__ == "__main__":
    import os
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    import torch.utils.data as data
    from tqdm.auto import tqdm
    import numpy as np
    from torch.utils.tensorboard.writer import SummaryWriter
    import dlutil.util as util
    import torch

    util.seed_everything(42)

    test_transform = image_to_numpy
    # For training, we add some augmentation. Networks are too powerful and would overfit.
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            image_to_numpy,
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
        num_workers=2,
        collate_fn=numpy_collate,
        persistent_workers=True,
    )
    val_loader = data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        collate_fn=numpy_collate,
        persistent_workers=True,
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        collate_fn=numpy_collate,
        num_workers=2,
    )

    trainer = ViTTrainer(
        lr=3e-4,
        embed_dim=256,
        hidden_dim=512,
        num_heads=8,
        num_layers=6,
        patch_size=4,
        num_channels=3,
        num_patches=64,
        num_classes=10,
        exmp_imgs=next(iter(train_loader))[0],
    )

    trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=180,
    )

    val_acc = trainer.eval_model(val_loader)
    test_acc = trainer.eval_model(test_loader)
    print(f"final val acc: {val_acc}, test_acc: {test_acc}")
