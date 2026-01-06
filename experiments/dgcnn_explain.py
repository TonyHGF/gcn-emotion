# experiments/dgcnn_explain.py

import os
import torch
import logging
from torch.utils.data import DataLoader

from models import DGCNN
from explainer import DGCNNExplainer
from data import SeedIVFeatureDataset


def run_dgcnn_explain(config: dict):
    """
    Post-hoc explanation for a trained DGCNN using feature-level EEG input.

    Expected config keys:
        - data_root
        - checkpoint_path
        - feature_key
        - num_classes
        - in_channels
        - num_electrodes
    """
    logger = logging.getLogger("DGCNN-Explain")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Explain] Using device: {device}")

    # --------------------------------------------------
    # 1. Dataset (feature-level)
    # --------------------------------------------------
    dataset = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key=config["feature_key"],
        sessions=[1, 2, 3],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # --------------------------------------------------
    # 2. Load trained model
    # --------------------------------------------------
    model = DGCNN(
        num_electrodes=config["num_electrodes"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
    ).to(device)

    ckpt_path = config["checkpoint_path"]
    logger.info(f"[Explain] Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # --------------------------------------------------
    # 3. One sample explanation
    # --------------------------------------------------
    x, y = next(iter(loader))
    x = x.to(device)  # (1, Ch, F)
    y = y.item()

    logger.info(f"[Explain] Explaining one sample, label={y}, shape={tuple(x.shape)}")

    explainer = DGCNNExplainer(
        model=model,
        num_nodes=config["num_electrodes"],
        steps=config.get("explain_steps", 300),
        lambda_sparse=config.get("lambda_sparse", 0.005),
        lambda_entropy=config.get("lambda_entropy", 0.1),
    ).to(device)

    edge_mask = explainer.explain(x)

    # --------------------------------------------------
    # 4. Save result
    # --------------------------------------------------
    os.makedirs("results", exist_ok=True)
    save_path = config.get("save_path", "results/edge_mask_sample.pt")
    torch.save(edge_mask.cpu(), save_path)

    logger.info(f"[Explain] Saved edge mask to {save_path}")
