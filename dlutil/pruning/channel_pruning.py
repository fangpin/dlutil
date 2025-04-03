import torch
import copy


def get_num_chans_to_keep(num_chans: int, sparsity: float) -> int:
    return round(num_chans * (1 - sparsity))


@torch.no_grad()
def channel_prune_last(
    backbone: torch.nn.Sequential, sparsity: list | float
) -> torch.nn.Sequential:
    """
    Perform channel pruning on the model.
    Pruning and keep the first k channels, k is calculated by sparsity.
    backbone: the submodule of the model to be pruned.
    The number of input channels of first kernel should not be prunned, because it's determined by the number of channels of input.
    Also the number of output channels of last kernel should not be preuned, becuase it will infect the output.
    """
    n_conv = len([m for m in backbone if isinstance(m, torch.nn.Conv2d)])
    if isinstance(sparsity, list):
        assert len(sparsity) == n_conv - 1
    else:
        sparsity = [sparsity] * (n_conv - 1)

    backbone = copy.deepcopy(backbone)
    all_convs = [m for m in backbone if isinstance(m, torch.nn.Conv2d)]
    all_bns = [m for m in backbone if isinstance(m, torch.nn.BatchNorm2d)]

    assert len(all_bns) == len(all_convs)

    # The number of input channels of first kernel should not be prunned, because it's determined by the number of channels of input.
    # Also the number of output channels of last kernel should not be preuned, becuase it will infect the output.
    for i, sp in enumerate(sparsity):
        prev_conv = all_convs[i]
        prev_bn = all_bns[i]
        next_conv = all_convs[i + 1]

        original_channels = prev_conv.out_channels
        n_keep = get_num_chans_to_keep(original_channels, sp)

        # pruning the output
        # the CNN kernel weight is formated as (output_channels, input_channels, kernel_width, kernel_height)
        prev_conv.weight.set_(prev_conv.weight.detach()[:n_keep])
        prev_bn.weight.set_(prev_bn.weight.detach()[:n_keep])
        prev_bn.bias.set_(prev_bn.bias.detach()[:n_keep])
        prev_bn.running_mean.set_(prev_bn.running_mean.detach()[:n_keep])
        prev_bn.running_var.set_(prev_bn.running_var.detach()[:n_keep])

        next_conv.weight.set_(
            next_conv.weight.detach()[:, :n_keep]
        )  # pruning the input
    return backbone


def input_channels_importance_l2(weight: torch.Tensor) -> torch.Tensor:
    """measure input channels' importance by l2 norm"""
    input_channels = weight.shape[1]
    importances = []
    for i in range(input_channels):
        w = weight.detach()[:, i]
        importance = (w**2).sum()
        importances.append(importance.view(1))
    return torch.cat(importances)


def channels_importance_rank(
    backbone: torch.nn.Sequential, measure=input_channels_importance_l2
):
    """
    Rerank the channels by importance.
    The importance is measured by `measure` func, default is input_channels_importance_l2.
    The input channels of first kernel should not be ranked, because it's determined by the number of channels of input.
    Also the output channels of last kernel should not be ranked, becuase it will infect the output.
    """
    backbone = copy.deepcopy(backbone)
    all_convs = [m for m in backbone if isinstance(m, torch.nn.Conv2d)]
    all_bns = [m for m in backbone if isinstance(m, torch.nn.BatchNorm2d)]
    for i in range(len(all_convs) - 1):
        prev_conv = all_convs[i]
        prev_bn = all_bns[i]
        next_conv = all_convs[i + 1]
        next_bn = all_convs[i + 1]

        importances = measure(next_bn.weight)
        sort_idx = torch.argsort(importances, descending=True)
        # prev output channel
        prev_conv.weight.copy_(torch.index_select(prev_conv.weight, 0, sort_idx))

        # batch norm
        for tenser_name in ["weight", "bias", "running_mean", "running_var"]:
            tensor_to_apply = getattr(prev_bn, tenser_name)
            tensor_to_apply.copy_(torch.index_select(tensor_to_apply, 0, sort_idx))

        # next input channel
        next_conv.weight.copy_(torch.index_select(next_conv.weight, 1, sort_idx))
    return backbone


def channel_prune_rank(
    backbone: torch.nn.Sequential,
    sparsity: list | float,
    measure=input_channels_importance_l2,
) -> torch.nn.Sequential:
    """
    Perform channel pruning on the model.
    Pruning and keep the k channels with most importance, k is calculated by sparsity. importance is measured by l2 norm
    backbone: the submodule of the model to be pruned.
    """
    backbone = channels_importance_rank(backbone, measure)
    return channel_prune_last(backbone, sparsity)
