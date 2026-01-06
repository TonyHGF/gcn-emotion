import torch
import torch.nn as nn
import torch.nn.functional as F


class DGCNNExplainer(nn.Module):
    """
    Edge-mask explainer for your DGCNN core model.

    Expected input:
        node_features: (B, N, F) where
            N = num_electrodes (e.g., 62)
            F = in_channels (e.g., 5 for de_LDS)
    """

    def __init__(
        self,
        model: nn.Module,
        num_nodes: int,
        lr: float = 0.01,
        lambda_sparse: float = 0.005,
        lambda_entropy: float = 0.1,
        steps: int = 300,
        symmetric_mask: bool = True,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        self.num_nodes = int(num_nodes)
        self.steps = int(steps)
        self.lambda_sparse = float(lambda_sparse)
        self.lambda_entropy = float(lambda_entropy)
        self.symmetric_mask = bool(symmetric_mask)

        # Mask logits -> sigmoid -> (0, 1)
        self.edge_mask_logits = nn.Parameter(torch.randn(self.num_nodes, self.num_nodes))
        self.optimizer = torch.optim.Adam([self.edge_mask_logits], lr=lr)

    @staticmethod
    def _entropy(mask: torch.Tensor) -> torch.Tensor:
        # mask in (0, 1)
        return -(
            mask * torch.log(mask + 1e-8) + (1.0 - mask) * torch.log(1.0 - mask + 1e-8)
        ).mean()

    def _build_mask(self) -> torch.Tensor:
        mask = torch.sigmoid(self.edge_mask_logits)

        if self.symmetric_mask:
            mask = 0.5 * (mask + mask.t())

        # Optional: remove self-loops if you want purely inter-electrode edges
        mask = mask * (1.0 - torch.eye(self.num_nodes, device=mask.device))

        return mask

    def explain(self, node_features: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        """
        Args:
            node_features: (B, N, F) e.g. (1, 62, 5)
            target_class: if None, uses model's predicted class for the given input

        Returns:
            edge_importance: (N, N) in [0, 1]
        """
        if node_features.dim() != 3:
            raise ValueError(f"Expected node_features with shape (B, N, F), got {tuple(node_features.shape)}")
        if node_features.shape[1] != self.num_nodes:
            raise ValueError(
                f"Expected N={self.num_nodes}, got node_features.shape[1]={node_features.shape[1]}"
            )

        with torch.no_grad():
            logits = self.model(node_features)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

        for _ in range(self.steps):
            self.optimizer.zero_grad()

            mask = self._build_mask()

            # Inject mask into adjacency
            masked_adj = self.model.adjacency * mask

            logits_masked = self._forward_with_custom_adjacency(node_features, masked_adj)

            # Keep the same prediction (maximize log-prob of target class)
            loss_pred = -F.log_softmax(logits_masked, dim=1)[0, target_class]

            # Regularize: sparse + low-entropy (push toward 0/1)
            loss_sparse = mask.mean()
            loss_entropy = self._entropy(mask)

            loss = loss_pred + self.lambda_sparse * loss_sparse + self.lambda_entropy * loss_entropy
            loss.backward()
            self.optimizer.step()

        return self._build_mask().detach()

    def _forward_with_custom_adjacency(self, node_features: torch.Tensor, new_adjacency: torch.Tensor) -> torch.Tensor:
        """
        Temporarily replace model.adjacency during forward.
        Works because your DGCNN uses self.adjacency inside forward().
        """
        original = self.model.adjacency.data.clone()
        try:
            self.model.adjacency.data = new_adjacency.data
            return self.model(node_features)
        finally:
            self.model.adjacency.data = original
