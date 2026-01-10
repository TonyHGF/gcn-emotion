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

from models import DGCNN
from data import SeedIVFeatureDataset
from utils import set_random_seed

from data.datasets import session_labels

def all_mix_split(config):
    # ---------- Dataset ----------
    dataset = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key="de_LDS",
        sessions=[1, 2, 3],)

    num_total = len(dataset)
    num_train = int(num_total * config["train_ratio"])
    num_val = int(num_total * config["val_ratio"])
    num_test = num_total - num_train - num_val

    train_set, val_set, test_set = random_split(
        dataset, [num_train, num_val, num_test])

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader, num_train, num_val, num_test

# Divide based on subject
def loso_split(config):
    pick = random.randint(1, 15)
    
    # Test: 1
    test_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_keys=["de_LDS"],
        sessions=[1, 2, 3],
        subjects=[pick])
    #Train: 14
    train_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_keys=["de_LDS"],
        sessions=[1, 2, 3],
        subjects=[i for i in range(1, 16) if (i != pick)])
    
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False)

    return train_loader, None, test_loader, len(train_set), 0, len(test_set)

def trial_split(config):
    train_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_keys=["de_LDS"],
        sessions=[1, 2, 3],
        split="train",
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        seed=config.get("seed", 42),
    )

    val_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_keys=["de_LDS"],
        sessions=[1, 2, 3],
        split="val",
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        seed=config.get("seed", 42),
    )

    test_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_keys=["de_LDS"],
        sessions=[1, 2, 3],
        split="test",
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        seed=config.get("seed", 42),
    )

    return (
        DataLoader(train_set, batch_size=config["batch_size"], shuffle=True),
        DataLoader(val_set, batch_size=config["batch_size"], shuffle=False),
        DataLoader(test_set, batch_size=config["batch_size"], shuffle=False),
        len(train_set),
        len(val_set),
        len(test_set),
    )


# ================= One Experiment =================
def run_one_experiment(exp_id: int, config: dict):
    logger = logging.getLogger(f"Exp-{exp_id}")
    logger.info(f"Running experiment {exp_id}")

    set_random_seed(exp_id)

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

    model = DGCNN(
        num_electrodes=config["num_electrodes"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"]
    ).to(device)

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