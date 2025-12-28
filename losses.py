from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


class B1ReLU(nn.Module):
    """
    Bias + ReLU with bias shape (1, 1, feature_dim).
    """

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.empty(1, 1, feature_dim))
        nn.init.zeros_(self.bias)
        self.activation = nn.ReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bias + input_tensor)


class B2ReLU(nn.Module):
    """
    Bias + ReLU with bias shape (1, num_nodes, feature_dim).
    """

    def __init__(self, num_nodes: int, feature_dim: int) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.empty(1, num_nodes, feature_dim))
        nn.init.zeros_(self.bias)
        self.activation = nn.ReLU()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.activation(self.bias + input_tensor)


class SparseL2Regularization(nn.Module):
    """
    L2 regularization term for a tensor (commonly used for adjacency matrix).
    """

    def __init__(self, l2_lambda: float) -> None:
        super().__init__()
        self.l2_lambda = float(l2_lambda)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        l2_norm = torch.norm(tensor, p=2)
        return self.l2_lambda * l2_norm


class NewSparseL2Regularization(nn.Module):
    """
    L2 regularization term over parameters of a module.
    """

    def __init__(self, l2_lambda: float) -> None:
        super().__init__()
        self.l2_lambda = float(l2_lambda)

    def forward(self, module: nn.Module) -> torch.Tensor:
        device = next(module.parameters()).device
        l2_reg = torch.tensor(0.0, device=device)
        for parameter in module.parameters():
            l2_reg = l2_reg + torch.norm(parameter)
        return l2_reg * self.l2_lambda