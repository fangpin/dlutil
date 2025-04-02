import torch


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


class FineGrainedPruner:
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
