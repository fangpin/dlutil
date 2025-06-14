import torch
import math
import time
from matplotlib import pyplot as plt

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB


def tensor_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the tensor sparsity.
    #zeros / #elements
    """
    return 1.0 - float(tensor.count_nonzero()) / tensor.numel()


def model_sparsity(model: torch.nn.Module) -> float:
    """
    calculate the model sparsity.
    #zeros / #elements
    """
    total_nonzeros = 0
    total_elements = 0
    for _, param in model.named_parameters():
        total_nonzeros += param.count_nonzero()
        total_elements += param.numel()
    return 1.0 - float(total_nonzeros) / total_elements


def num_params(model: torch.nn.Module, non_zero_only=False) -> int:
    """
    calculate the number of parameters in the model.
    """
    total_params = 0
    for _, param in model.named_parameters():
        if non_zero_only:
            total_params += int(param.count_nonzero())
        else:
            total_params += param.numel()
    return total_params


def model_size(model: torch.nn.Module, data_with=32, non_zero_only=False) -> int:
    """
    calculate the model size in bits.
    """
    return data_with * num_params(model, non_zero_only)


def model_latency(
    model: torch.nn.Module,
    dummy_input: torch.Tensor,
    n_warmup=20,
    n_test=100,
    backend="cpu",
) -> float:
    """measure model latency in ms"""
    if backend == "cpu":
        model = model.to("cpu")
    else:
        model = model.to("gpu")
    model.eval()
    for _ in range(n_warmup):
        _ = model(dummy_input)
    t1 = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)
    t2 = time.time()
    return 1000.0 * (t2 - t1) / n_test


def plot_weights_distribution(model, bins=256, non_zero_only=False):
    num_weight_group = len(model.named_parameters())
    fig, axes = plt.subplots(4, math.ceil(num_weight_group / 3), figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            if non_zero_only:
                param_cpu = param.detach().view(-1).cpu()
                param_cpu = param_cpu[param_cpu != 0].view(-1)
                ax.hist(param_cpu, bins=bins, density=True, color="blue", alpha=0.5)
            else:
                ax.hist(
                    param.detach().view(-1).cpu(),
                    bins=bins,
                    density=True,
                    color="blue",
                    alpha=0.5,
                )
            ax.set_xlabel(name)
            ax.set_ylabel("density")
            plot_index += 1
    fig.suptitle("Histogram of Weights")
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()


def plot_num_parameters_distribution(model: torch.nn.Module):
    name_parma_num = dict()
    for name, param in model.named_parameters():
        if param.dim() > 1:
            name_parma_num[name] = param.numel()
    fig = plt.figure(figsize=(8, 6))
    plt.grid(axis="y")
    plt.bar(list(name_parma_num.keys()), list(name_parma_num.values()))
    plt.title("parameters distribution")
    plt.ylabel("number of parameters")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()
