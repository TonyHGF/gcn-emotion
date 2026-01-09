# main.py

import os
import sys
import argparse
import logging
import numpy as np

from experiments.run_dgcnn_seediv import run_one_experiment
from experiments.dgcnn_explain import run_dgcnn_explain


def build_parser():
    parser = argparse.ArgumentParser(
        description="Main entry for DGCNN training and explanation"
    )

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "explain"],
        help="Run mode: train or explain"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="dgcnn",
        choices=["dgcnn"],
        help="Model type"
    )

    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory of dataset"
    )

    # -------- Training arguments --------
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--exp_times", type=int, default=5)
    parser.add_argument("--split", type=str, default="all") # "all", "loso", "trial"
    parser.add_argument("--checkpoints_folder", type=str)

    # -------- Explain arguments --------
    parser.add_argument("--feature_key", type=str, default="de_LDS")
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--num_electrodes", type=int, default=62)
    parser.add_argument("--in_channels", type=int, default=5)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument(
        "--test_sessions",
        type=int,
        nargs="+",
        default=[1, 2, 3]
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save all outputs of this run"
    )

    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s:%(name)s: %(message)s",
        stream=sys.stdout,
    )

    parser = build_parser()
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "train" and args.model == "dgcnn":
        config = {
            "data_root": args.data_root,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_epochs": args.num_epochs,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "num_electrodes": args.num_electrodes,
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
            "checkpoints_folder": args.checkpoints_folder,
            "output_dir": args.output_dir,
            "split": args.split
        }

        all_test_acc = []

        for exp_id in range(args.exp_times):
            acc = run_one_experiment(exp_id + 1, config)
            all_test_acc.append(acc)

        logging.info("========== Final Summary ==========")
        logging.info(f"Experiment Times: {args.exp_times}")
        logging.info(f"Test Accuracies: {all_test_acc}")
        logging.info(
            f"Mean = {np.mean(all_test_acc):.4f}, "
            f"Std = {np.std(all_test_acc):.4f}"
        )

    elif args.mode == "explain" and args.model == "dgcnn":
        if args.checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided in explain mode")

        explain_config = {
            "data_root": args.data_root,
            "feature_key": args.feature_key,
            "checkpoint_path": args.checkpoint_path,
            "num_electrodes": args.num_electrodes,
            "in_channels": args.in_channels,
            "num_classes": args.num_classes,
            "test_sessions": args.test_sessions,
            "output_dir": args.output_dir,
        }

        run_dgcnn_explain(explain_config)

    else:
        raise ValueError(f"Unsupported mode={args.mode}, model={args.model}")


if __name__ == "__main__":
    main()
