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

from data import SeedIVFeatureDataset

def all_mix_split(config):
    dataset = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_keys=["de_LDS"],
        sessions=[1, 2, 3],
    )

    num_total = len(dataset)
    num_train = int(num_total * config["train_ratio"])
    num_val = int(num_total * config["val_ratio"])
    num_test = num_total - num_train - num_val

    train_set, val_set, test_set = random_split(
        dataset, [num_train, num_val, num_test]
    )

    return (
        DataLoader(train_set, batch_size=config["batch_size"], shuffle=True),
        DataLoader(val_set, batch_size=config["batch_size"], shuffle=False),
        DataLoader(test_set, batch_size=config["batch_size"], shuffle=False),
        len(train_set),
        len(val_set),
        len(test_set),
    )

def loso_split(config):
    pick = random.randint(1, 15)
    
    train_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_keys=["de_LDS"],
        sessions=[1, 2, 3],
        subjects=[i for i in range(1, 16) if (i != pick)],
    )

    test_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_keys=["de_LDS"],
        sessions=[1, 2, 3],
        subjects=[pick],
    )

    return (
        DataLoader(train_set, batch_size=config["batch_size"], shuffle=True),
        None,
        DataLoader(test_set, batch_size=config["batch_size"], shuffle=False),
        len(train_set),
        0,
        len(test_set),
    )

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