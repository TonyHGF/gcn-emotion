# experiments/dgcnn_explain.py

import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader

from models import DGCNN
from explainer import DGCNNInterpreter
from data import SeedIVFeatureDataset


def run_dgcnn_explain(config: dict):
    """
    Post-hoc explanation for DGCNN using gradient-based saliency.

    This function performs group-level attribution analysis
    on correctly predicted samples for each class.

    Expected config keys:
        - data_root
        - checkpoint_path
        - feature_key
        - num_classes
        - in_channels
        - num_electrodes
        - test_sessions (optional)
        - save_path (optional)
    """
    logger = logging.getLogger("DGCNN-Explain")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Explain] Using device: {device}")

    # --------------------------------------------------
    # 1. Dataset (test split only)
    # --------------------------------------------------
    # Use unseen sessions for explanation to avoid leakage
    dataset = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key=config["feature_key"],
        sessions=config.get("test_sessions", [3]),
    )

    # Disable shuffle for reproducibility
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    logger.info(f"[Explain] Loaded dataset with {len(dataset)} samples.")

    # --------------------------------------------------
    # 2. Load trained DGCNN model
    # --------------------------------------------------
    model = DGCNN(
        num_electrodes=config["num_electrodes"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
    ).to(device)

    ckpt_path = config["checkpoint_path"]
    logger.info(f"[Explain] Loading checkpoint from {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # --------------------------------------------------
    # 3. Initialize gradient-based interpreter
    # --------------------------------------------------
    explainer = DGCNNInterpreter(model, device=device)

    results = {}
    num_classes = config["num_classes"]

    logger.info("[Explain] Starting group-level saliency analysis...")

    # --------------------------------------------------
    # 4. Loop over all classes
    # --------------------------------------------------
    for class_idx in range(num_classes):
        logger.info(f"[Explain] Analyzing class {class_idx}")

        # Compute group-averaged node and edge importance
        avg_node, avg_edge, sample_count = explainer.explain_group(
            loader, class_idx
        )

        if avg_node is not None:
            results[class_idx] = {
                "node_importance": avg_node,
                "edge_importance": avg_edge,
                "sample_count": sample_count,
            }
            logger.info(
                f"[Explain] {sample_count} correctly predicted samples used for class {class_idx}"
            )
        else:
            logger.warning(
                f"[Explain] No correct predictions found for class {class_idx}"
            )

    # --------------------------------------------------
    # 5. Save explanation results
    # --------------------------------------------------
    os.makedirs("results", exist_ok=True)
    save_path = config.get("save_path", "results/global_explanation.pt")

    torch.save(results, save_path)
    logger.info(f"[Explain] Global explanation saved to {save_path}")

    # --------------------------------------------------
    # 6. (Optional) Save learned static adjacency
    # --------------------------------------------------
    # This adjacency is global and not sample-dependent
    if hasattr(model, "adjacency"):
        global_adj = (
            model.activation(model.adjacency + model.adjacency_bias)
            .detach()
            .cpu()
        )
        torch.save(global_adj, "results/learned_static_adjacency.pt")
        logger.info("[Explain] Saved learned static adjacency")
