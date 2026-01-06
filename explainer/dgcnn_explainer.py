import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DGCNNInterpreter:
    def __init__(self, model, device="cuda"):
        """
        Initialize the interpreter.
        model: trained DGCNN model
        device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.model.eval()  # must use eval mode to disable dropout
        self.device = device

    def _compute_single_sample_saliency(self, sample, target_class_idx):
        """
        Compute gradient-based saliency for one sample.
        sample shape: (1, num_electrodes, num_features)
        """
        # Prepare input
        input_tensor = sample.clone().detach().to(self.device)
        input_tensor.requires_grad = True

        # Clear old gradients
        self.model.zero_grad()

        # Forward pass
        logits = self.model(input_tensor)

        # Select target class score
        score = logits[0, target_class_idx]

        # Backward pass
        score.backward()

        # -----------------------------
        # Node importance
        # -----------------------------
        # Sum absolute gradients over feature dimension
        node_saliency = (
            input_tensor.grad.abs()
            .sum(dim=-1)
            .squeeze()
            .cpu()
            .numpy()
        )

        # -----------------------------
        # Edge importance
        # -----------------------------
        # Use gradient of learnable adjacency matrix
        if self.model.adjacency.grad is not None:
            edge_saliency = self.model.adjacency.grad.abs().cpu().numpy()
        else:
            edge_saliency = np.zeros(
                (self.model.num_electrodes, self.model.num_electrodes)
            )

        return node_saliency, edge_saliency

    def explain_group(self, test_loader, target_class_idx):
        """
        Compute average importance for one class over the test set.
        """
        print(f"Analyzing target class: {target_class_idx}")

        node_saliency_list = []
        edge_saliency_list = []
        count = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Process samples one by one
            for i in range(len(inputs)):
                sample = inputs[i].unsqueeze(0)
                label = labels[i].item()

                # Only analyze target class
                if label != target_class_idx:
                    continue

                # Check prediction correctness
                with torch.no_grad():
                    pred = self.model(sample).argmax(dim=1).item()

                if pred == label:
                    n_sal, e_sal = self._compute_single_sample_saliency(
                        sample, target_class_idx
                    )
                    node_saliency_list.append(n_sal)
                    edge_saliency_list.append(e_sal)
                    count += 1

        if count == 0:
            print("No correctly predicted samples found.")
            return None, None

        print(f"Used {count} correctly predicted samples.")

        # Average over samples
        avg_node_importance = np.mean(node_saliency_list, axis=0)
        avg_edge_importance = np.mean(edge_saliency_list, axis=0)

        return avg_node_importance, avg_edge_importance, count

    def visualize_result(
        self,
        avg_node,
        avg_edge,
        class_name="Happy",
        save_path="dgcnn_explain.png"
    ):
        """
        Save node and edge importance figures to disk.
        """
        plt.figure(figsize=(12, 5))

        # Node importance
        plt.subplot(1, 2, 1)
        plt.bar(range(len(avg_node)), avg_node)
        plt.title(f"{class_name} - Node Importance")
        plt.xlabel("Electrode Index")
        plt.ylabel("Importance")

        # Edge importance
        plt.subplot(1, 2, 2)
        sns.heatmap(avg_edge, cmap="Reds")
        plt.title(f"{class_name} - Edge Importance")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

