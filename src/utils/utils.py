from torch.nn import Module
from torch import Tensor, FloatTensor, pow, sin, cos, arange
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model:nn.Module)->int:
    """
    Count the number of parameters in a model.
    
    Args:
    - model: nn.Module, the model to count parameters.
    
    Returns:
    - int, the number of parameters in the model.
    """
    num_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Number of parameters: {num_params}")
    return num_params


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def is_symmetric(matrix, tol=1e-8):
    return torch.allclose(matrix, matrix.T, atol=tol)


class weighted_MSELoss(Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs, targets, weights):
        diff = (inputs - targets) ** 2
        weights = weights.to(diff.device)
        if weights.shape != diff.shape:
            if weights.numel() == diff.numel():
                weights = weights.reshape(diff.shape)
            else:
                while weights.dim() < diff.dim():
                    weights = weights.unsqueeze(0)
                try:
                    weights = weights.expand(diff.shape)
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Incompatible weight shape {tuple(weights.shape)} for target shape {tuple(diff.shape)}"
                    ) from exc
        return diff * weights
