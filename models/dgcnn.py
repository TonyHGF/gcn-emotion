# We utilize DGCNN to implement our GCN.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .graph_utils import laplacian, GraphConv
from .losses import B1ReLU, B2ReLU, SparseL2Regularization, NewSparseL2Regularization
from utils import FeatureExtractorConfig


class DGCNN(nn.Module):
    """
    DGCNN core model.

    Expected input:
        node_features: (B, num_electrodes, in_channels)

    Notes:
        - Adjacency is learned as a parameter.
        - Laplacian is computed from relu(adj + adj_bias).
    """

    def __init__(
        self,
        num_electrodes: int = 62,
        in_channels: int = 5,
        num_classes: int = 3,
        chebyshev_order_k: int = 2,
        relu_type: int = 1,
        layers: Optional[List[int]] = None,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()

        self.num_electrodes = int(num_electrodes)
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.chebyshev_order_k = int(chebyshev_order_k)
        self.relu_type = int(relu_type)
        self.dropout_rate = float(dropout_rate)

        if layers is None:
            # Keep the behavior similar to the code you provided
            if self.num_electrodes == 62:
                layers = [64]
            elif self.num_electrodes == 32:
                layers = [128]
            else:
                layers = [64]

        self.layers = layers

        self.graph_convolutions = nn.ModuleList()
        self.graph_convolutions.append(
            GraphConv(self.chebyshev_order_k, self.in_channels, self.layers[0])
        )
        for layer_index in range(len(self.layers) - 1):
            self.graph_convolutions.append(
                GraphConv(self.chebyshev_order_k, self.layers[layer_index], self.layers[layer_index + 1])
            )

        self.fc1 = nn.Linear(self.num_electrodes * self.layers[-1], 256, bias=True)
        self.fc2 = nn.Linear(256, self.num_classes, bias=True)

        self.adjacency = nn.Parameter(torch.empty(self.num_electrodes, self.num_electrodes))
        self.adjacency_bias = nn.Parameter(torch.empty(1))

        self.activation = nn.ReLU(inplace=True)

        self.bias_relus = nn.ModuleList()
        if self.relu_type == 1:
            for feature_dim in self.layers:
                self.bias_relus.append(B1ReLU(feature_dim))
        elif self.relu_type == 2:
            for feature_dim in self.layers:
                self.bias_relus.append(B2ReLU(self.num_electrodes, feature_dim))
        else:
            raise ValueError("relu_type must be 1 (B1ReLU) or 2 (B2ReLU)")

        self.dropout = nn.Dropout(p=self.dropout_rate)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.adjacency)
        nn.init.trunc_normal_(self.adjacency_bias, mean=0.0, std=0.1)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: (B, N, F_in)

        Returns:
            logits: (B, num_classes)
        """
        adjacency_matrix = self.activation(self.adjacency + self.adjacency_bias)  # (N, N)
        laplacian_matrix = laplacian(adjacency_matrix)  # (N, N)

        features = node_features
        for layer_index in range(len(self.layers)):
            features = self.graph_convolutions[layer_index](features, laplacian_matrix)
            features = self.dropout(features)
            features = self.bias_relus[layer_index](features)

        features = features.reshape(features.shape[0], -1)  # (B, N * F_last)
        features = self.dropout(features)
        features = self.fc1(features)
        features = self.dropout(features)
        logits = self.fc2(features)
        return logits


class DGCNNAdapter(nn.Module):
    """
    Adapter for your current loader output shape: (B, 1, Ch, T).

    It performs:
        1) squeeze dim=1 -> (B, Ch, T)
        2) channel-wise feature extraction -> (B, Ch, F)
        3) forward through DGCNN -> (B, num_classes)

    This keeps DGCNN core unchanged and isolates shape/feature logic here.
    """

    def __init__(
        self,
        num_electrodes: int = 62,
        num_classes: int = 4,
        chebyshev_order_k: int = 2,
        relu_type: int = 1,
        layers: Optional[List[int]] = None,
        dropout_rate: float = 0.5,
        feature_extractor_config: Optional[FeatureExtractorConfig] = None,
    ) -> None:
        super().__init__()

        self.feature_extractor_config = feature_extractor_config or FeatureExtractorConfig(mode="variance")

        # Determine in_channels based on feature mode
        in_channels = self._infer_in_channels(self.feature_extractor_config.mode)

        self.dgcnn = DGCNN(
            num_electrodes=num_electrodes,
            in_channels=in_channels,
            num_classes=num_classes,
            chebyshev_order_k=chebyshev_order_k,
            relu_type=relu_type,
            layers=layers,
            dropout_rate=dropout_rate,
        )

    @staticmethod
    def _infer_in_channels(mode: str) -> int:
        if mode == "variance":
            return 1
        if mode == "mean_var":
            return 2
        if mode == "identity_time":
            # This will be overwritten dynamically if you really use it.
            # Keeping a placeholder here; for identity_time you should set DGCNN.in_channels to T explicitly.
            return 1
        raise ValueError(f"Unsupported feature extractor mode: {mode}")

    def extract_features(self, signal_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signal_tensor: (B, Ch, T)

        Returns:
            node_features: (B, Ch, F)
        """
        mode = self.feature_extractor_config.mode

        if mode == "variance":
            variance_feature = signal_tensor.var(dim=-1, unbiased=False).unsqueeze(-1)  # (B, Ch, 1)
            return variance_feature

        if mode == "mean_var":
            mean_feature = signal_tensor.mean(dim=-1).unsqueeze(-1)  # (B, Ch, 1)
            variance_feature = signal_tensor.var(dim=-1, unbiased=False).unsqueeze(-1)  # (B, Ch, 1)
            return torch.cat([mean_feature, variance_feature], dim=-1)  # (B, Ch, 2)

        if mode == "identity_time":
            # (B, Ch, T) -> treat T as feature dim
            return signal_tensor

        raise ValueError(f"Unsupported feature extractor mode: {mode}")

    def forward(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            batch_tensor: (B, 1, Ch, T) or (B, Ch, T)

        Returns:
            logits: (B, num_classes)
        """
        if batch_tensor.dim() == 4:
            # (B, 1, Ch, T) → (B, Ch, T)
            signal_tensor = batch_tensor.squeeze(1)
        elif batch_tensor.dim() == 3:
            # already (B, Ch, T)
            signal_tensor = batch_tensor
        else:
            raise ValueError(
                f"Expected input shape (B, 1, Ch, T) or (B, Ch, T), "
                f"got shape: {tuple(batch_tensor.shape)}"
            )

        node_features = self.extract_features(signal_tensor)  # (B, Ch, F)
        logits = self.dgcnn(node_features)
        return logits
