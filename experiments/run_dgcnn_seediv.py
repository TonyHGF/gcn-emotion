import os
import sys
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from models import DGCNN, PGCN
from data import SeedIVFeatureDataset, all_mix_split, trial_split, loso_split
from data.datasets import session_labels


# ================= One Experiment =================
def run_one_experiment(exp_id: int, config: dict):
    logger = logging.getLogger(f"Exp-{exp_id}")
    logger.info(f"Running experiment {exp_id}")

    if config["split"] == "all":
        split = all_mix_split
    elif config["split"] == "loso":
        split = loso_split
    else:
        split = trial_split
    train_loader, val_loader, test_loader, num_train, num_val, num_test = split(config)
    
    num_total = num_train + num_val + num_test

    logger.info("-" * 30)
    logger.info("Experiment Configuration:")
    for key, value in config.items():
        logger.info(f"  {key:<15}: {value}")
    logger.info("-" * 30)

    logger.info(f"Dataset Summary for Experiment {exp_id}:")
    logger.info(f"  Total Samples : {num_total}")
    logger.info(f"  Training Set  : {num_train} samples")
    logger.info(f"  Validation Set: {num_val} samples")
    logger.info(f"  Testing Set   : {num_test} samples")
    logger.info("-" * 30)

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = None
    if config["model"] == "dgcnn":
        model = DGCNN(
            num_electrodes=config["num_electrodes"],
            in_channels=config["in_channels"],
            num_classes=config["num_classes"]
        ).to(device)
    elif config["model"] == "pgcn":
        model = PGCN(
            num_electrodes=config["num_electrodes"],
            in_channels=config["in_channels"],
            num_classes=config["num_classes"],
            dropout_rate=config.get("dropout", 0.5),
            lr=0.1
        ).to(device)
    else:
        return None

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    # ---------- Training ----------
    history = {
        "train_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    os.makedirs(config["checkpoints_folder"], exist_ok=True)
    ckpt_path = os.path.join(config["checkpoints_folder"], f"best_model_exp{exp_id}.pth")

    best_train_loss = float('inf')
    for epoch in range(config["num_epochs"]):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["train_loss"].append(avg_loss)

        # ----- Validation -----
        if val_loader is not None:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)

                    preds = model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)

            val_acc = correct / total
            history["val_acc"].append(val_acc)

            logger.info(
                f"Epoch [{epoch+1}/{config['num_epochs']}] "
                f"Loss={avg_loss:.4f} ValAcc={val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved best model (ValAcc={best_val_acc:.4f})")
        else:
            logger.info(
                f"Epoch [{epoch+1}/{config['num_epochs']}] "
                f"Loss={avg_loss:.4f}"
            )
            if avg_loss < best_train_loss:
                best_train_loss = avg_loss
                torch.save(model.state_dict(), ckpt_path)
                logger.info(f"Saved model based on Min Train Loss ({avg_loss:.4f})")

    # ---------- Testing ----------
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    logger.info(f"Experiment {exp_id} | Test Accuracy = {test_acc:.4f}")

    # ---------- Plot ----------
    fig_path = os.path.join(config["output_dir"], f"training_curves_exp{exp_id}.png")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"])
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history["val_acc"], label="Val Acc")
    plt.axhline(
        y=test_acc,
        linestyle="--",
        label=f"Test Acc={test_acc:.2f}",
    )
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    logger.info(f"Saved plot: {fig_path}")

    return test_acc