from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn


def laplacian(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute the normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}.

    Args:
        adjacency_matrix: (N, N) non-negative adjacency matrix.

    Returns:
        laplacian_matrix: (N, N) normalized Laplacian matrix.
    """
    degree_vector = torch.sum(adjacency_matrix, dim=1)  # (N,)
    degree_inverse_sqrt = 1.0 / torch.sqrt(degree_vector + 1e-5)  # (N,)
    degree_matrix = torch.diag_embed(degree_inverse_sqrt)  # (N, N)
    identity_matrix = torch.eye(degree_matrix.shape[0], device=adjacency_matrix.device)
    laplacian_matrix = identity_matrix - degree_matrix @ adjacency_matrix @ degree_matrix
    return laplacian_matrix


class GraphConv(nn.Module):
    """
    Graph convolution based on Chebyshev polynomials.

    Input:
        node_features: (batch_size, num_nodes, in_channels)
        laplacian_matrix: (num_nodes, num_nodes)

    Output:
        transformed_features: (batch_size, num_nodes, out_channels)
    """

    def __init__(self, chebyshev_order_k: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if chebyshev_order_k < 1:
            raise ValueError("chebyshev_order_k must be >= 1")

        self.chebyshev_order_k = chebyshev_order_k
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.empty(chebyshev_order_k * in_channels, out_channels))
        nn.init.xavier_uniform_(self.weight)

    def chebyshev_polynomial(
        self, node_features: torch.Tensor, laplacian_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Chebyshev polynomial components up to order K-1.

        Args:
            node_features: (B, N, F)
            laplacian_matrix: (N, N)

        Returns:
            chebyshev_components: (B, K, N, F)
        """
        batch_size, num_nodes, feature_dim = node_features.shape
        ones_component = torch.ones(batch_size, num_nodes, feature_dim, device=node_features.device)

        if self.chebyshev_order_k == 1:
            return ones_component.unsqueeze(1)

        if self.chebyshev_order_k == 2:
            first_order = laplacian_matrix @ node_features
            return torch.cat((ones_component.unsqueeze(1), first_order.unsqueeze(1)), dim=1)

        # K >= 3
        t_k_minus_two = node_features  # T_0 (as used in the provided code)
        t_k_minus_one = laplacian_matrix @ node_features  # T_1
        components = torch.cat(
            (ones_component.unsqueeze(1), t_k_minus_two.unsqueeze(1), t_k_minus_one.unsqueeze(1)),
            dim=1,
        )

        for order_index in range(3, self.chebyshev_order_k):
            t_k = 2.0 * (laplacian_matrix @ t_k_minus_one) - t_k_minus_two
            components = torch.cat((components, t_k.unsqueeze(1)), dim=1)
            t_k_minus_two, t_k_minus_one = t_k_minus_one, t_k

        return components

    def forward(self, node_features: torch.Tensor, laplacian_matrix: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (B, N, F_in)
            laplacian_matrix: (N, N)

        Returns:
            output_features: (B, N, F_out)
        """
        chebyshev_components = self.chebyshev_polynomial(node_features, laplacian_matrix)  # (B, K, N, F)
        chebyshev_components = chebyshev_components.permute(0, 2, 3, 1)  # (B, N, F, K)
        chebyshev_components = chebyshev_components.flatten(start_dim=2)  # (B, N, F*K)
        output_features = chebyshev_components @ self.weight  # (B, N, F_out)
        return output_features
