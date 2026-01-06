# main.py

import os
import sys
import logging
import numpy as np

from experiments.run_dgcnn_seediv import run_one_experiment
from experiments.dgcnn_explain import run_dgcnn_explain


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        stream=sys.stdout,
    )

    # ================= USER CONTROL =================
    mode = "explain"      # "train" or "explain"
    model = "dgcnn"     # currently only dgcnn
    # =================================================

    if mode == "train" and model == "dgcnn":
        config = {
            "data_root": "/public/home/hugf2022/emotion/seediv/eeg_feature_smooth",
            "batch_size": 64,
            "lr": 1e-3,
            "num_epochs": 50,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
        }

        exp_times = 5
        all_test_acc = []

        for exp_id in range(exp_times):
            acc = run_one_experiment(exp_id + 1, config)
            all_test_acc.append(acc)

        logging.info("========== Final Summary ==========")
        logging.info(f"Experiment Times: {exp_times}")
        logging.info(f"Test Accuracies: {all_test_acc}")
        logging.info(
            f"Mean = {np.mean(all_test_acc):.4f}, "
            f"Std = {np.std(all_test_acc):.4f}"
        )

    elif mode == "explain" and model == "dgcnn":
        explain_config = {
            "data_root": "/public/home/hugf2022/emotion/seediv/eeg_feature_smooth",
            "feature_key": "de_LDS",
            "checkpoint_path": "checkpoints/best_model_exp1.pth",
            "num_electrodes": 62,
            "in_channels": 5,
            "num_classes": 4,
            "test_sessions": [1, 2, 3], 
            "save_path": "results/dgcnn_global_explanation.pt",
        }


        run_dgcnn_explain(explain_config)

    else:
        raise ValueError(f"Unsupported mode={mode}, model={model}")


if __name__ == "__main__":
    main()
