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

from model import DGCNN
from datasets import SeedIVFeatureDataset


# ================= Random Seed =================
def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================= One Experiment =================
def run_one_experiment(seed: int, config: dict):
    logger = logging.getLogger(f"Seed-{seed}")
    logger.info(f"Running experiment with seed = {seed}")

    set_random_seed(seed)

    # ---------- Dataset ----------
    dataset = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key="de_LDS",
        sessions=[1, 2, 3],
    )

    num_total = len(dataset)
    num_train = int(num_total * config["train_ratio"])
    num_val = int(num_total * config["val_ratio"])
    num_test = num_total - num_train - num_val

    train_set, val_set, test_set = random_split(
        dataset, [num_train, num_val, num_test]
    )

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False
    )
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False
    )

    # ---------- Model ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = DGCNN(
        num_electrodes=62,
        in_channels=5,
        num_classes=4,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # ---------- Training ----------
    history = {
        "train_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0.0
    ckpt_path = f"checkpoints/best_model_seed{seed}.pth"

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
    logger.info(f"Seed {seed} | Test Accuracy = {test_acc:.4f}")

    # ---------- Plot ----------
    fig_path = f"results/training_curves_seed{seed}.png"

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


# ================= Main =================
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        stream=sys.stdout,
    )

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    config = {
        "data_root": "/public/home/hugf2022/emotion/seediv/eeg_feature_smooth",
        "batch_size": 64,
        "lr": 1e-3,
        "num_epochs": 50,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
    }

    # ===== Control number of runs here =====
    seeds = [0, 1, 2, 3, 4]

    all_test_acc = []

    for seed in seeds:
        acc = run_one_experiment(seed, config)
        all_test_acc.append(acc)

    logging.info("========== Final Summary ==========")
    logging.info(f"Seeds: {seeds}")
    logging.info(f"Test Accuracies: {all_test_acc}")
    logging.info(
        f"Mean = {np.mean(all_test_acc):.4f}, "
        f"Std = {np.std(all_test_acc):.4f}"
    )


if __name__ == "__main__":
    main()
