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

from experiments.run_dgcnn_seediv import run_one_experiment


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
    exp_times = 5

    all_test_acc = []

    for exp_id in range(exp_times):
        acc = run_one_experiment(exp_id+1, config)
        all_test_acc.append(acc)

    logging.info("========== Final Summary ==========")
    logging.info(f"Experiment Times: {exp_times}")
    logging.info(f"Test Accuracies: {all_test_acc}")
    logging.info(
        f"Mean = {np.mean(all_test_acc):.4f}, "
        f"Std = {np.std(all_test_acc):.4f}"
    )


if __name__ == "__main__":
    main()
