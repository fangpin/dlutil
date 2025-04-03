import torch
import numpy as np
import math
from tqdm.auto import tqdm
from matplotlib import pyplot as plt


def tensor_pruning_l1(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """
    calculate the mask that's able to prune the tensor with specific sparsity
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    num_zeros = round(num_elements * sparsity)
    importance = tensor.abs()
    threshold, _ = importance.flatten().kthvalue(num_zeros)
    mask = torch.gt(importance, threshold)

    return mask


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> float:
    raise RuntimeError("not implemented")


@torch.no_grad()
def sensitivity_scan(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    scan_start=0.1,
    scan_end=1.0,
    scan_step=0.1,
    eval_f=evaluate,
):
    """
    calculate the sensitivity of each layer's sparsity and model accuracy.
    """
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    named_conv_weights = [
        (name, param) for (name, param) in model.named_parameters() if param.dim() > 1
    ]
    accuracies = []
    for i, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        accuracy = []
        for sparsity in tqdm(
            sparsities, desc=f"scanning {i}/{len(named_conv_weights)} weight - {name}"
        ):
            mask = tensor_pruning_l1(param.detach(), float(sparsity))
            # prune the param
            param.mul_(mask)
            # test the model accuracy
            acc = eval_f(model, dataloader)
            print(f"sparsity: {sparsity}, accuracy: {acc}")
            # restore the model
            param.copy_(param_clone.detach())
            accuracy.append(acc)
        accuracies.append(accuracy)
    return sparsities, accuracies


def plot_sensitivity_scan(
    model: torch.nn.Module, sparsities, accuracies, dense_model_accuracy
):
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    fig, axes = plt.subplots(3, int(math.ceil(len(accuracies) / 3)), figsize=(15, 8))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            curve = ax.plot(sparsities, accuracies[plot_index])
            line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
            ax.set_ylim(80, 95)
            ax.set_title(name)
            ax.set_xlabel("sparsity")
            ax.set_ylabel("top-1 accuracy")
            ax.legend(
                [
                    "accuracy after pruning",
                    f"{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy",
                ]
            )
            ax.grid(axis="x")
            plot_index += 1
    fig.suptitle("Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity")
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


class FineGrainedPruner:
    """prune the model with fine-grained specific sparsity for each layer"""

    def __init__(self, model: torch.nn.Module, sparsity_dict: dict[str, float]):
        self.masks = FineGrainedPruner._masks(model, sparsity_dict)
        self.model = model

    @staticmethod
    def _masks(
        model: torch.nn.Module, sparsity_dict: dict[str, float]
    ) -> dict[str, torch.Tensor]:
        masks = dict()
        for name, params in model.named_parameters():
            if (
                params.dim() > 1 and name in sparsity_dict
            ):  # only prune conv and fc layers
                masks[name] = tensor_pruning_l1(params, sparsity_dict[name])
        return masks

    @torch.no_grad()
    def apply(self):
        for name, params in self.model.named_parameters():
            if name in self.masks:
                params.mul_(self.masks[name])
