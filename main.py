import os
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model import DGCNN
from datasets import SeedIVFeatureDataset


def main():
    # ================= Logging =================
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger("Main")

    # ================= Paths =================
    data_root = "/public/home/hugf2022/emotion/seediv/eeg_feature_smooth"
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # ================= Dataset =================
    logger.info("Loading SEED-IV feature dataset...")

    dataset = SeedIVFeatureDataset(
        root=data_root,
        feature_key="de_LDS",
        sessions=[1, 2, 3],
    )

    logger.info(f"Total segments: {len(dataset)}")

    # ================= Split =================
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    num_total = len(dataset)
    num_train = int(num_total * train_ratio)
    num_val = int(num_total * val_ratio)
    num_test = num_total - num_train - num_val

    train_set, val_set, test_set = random_split(
        dataset, [num_train, num_val, num_test]
    )

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

    logger.info(
        f"Split: Train={num_train}, Val={num_val}, Test={num_test}"
    )

    # ================= Model =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = DGCNN(
        num_electrodes=62,
        in_channels=5,
        num_classes=4,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ================= Training =================
    num_epochs = 50
    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "val_acc": [],
    }

    logger.info("Start training...")
    for epoch in range(num_epochs):
        # -------- Train --------
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

        # -------- Validation --------
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        history["val_acc"].append(val_acc)

        logger.info(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss={avg_loss:.4f} "
            f"Val Acc={val_acc:.4f}"
        )

        # -------- Checkpoint --------
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pth",
            )
            logger.info(
                f"Best model saved (Val Acc={best_val_acc:.4f})"
            )

    # ================= Testing =================
    logger.info("Start testing...")

    best_model_path = "checkpoints/best_model.pth"
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logger.info("Loaded best model checkpoint.")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    test_acc = correct / total
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")

    # ================= Curves =================
    logger.info("Saving training curves...")

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
    plt.savefig("results/training_curves.png")
    logger.info("Saved results/training_curves.png")


if __name__ == "__main__":
    main()
