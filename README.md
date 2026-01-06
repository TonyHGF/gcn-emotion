# Aligning GCN Attribution Maps with EEG Microstate Dynamics for Emotion Recognition

> ShanghaiTech BME2127 final project
>
> Author: Gangfeng Hu, Zhiyu Yang, Xinyu Zhao, Tianyi Zhao, Yueying Wang

## Overview

This project investigates interpretable EEG-based emotion recognition using graph convolutional networks (GCNs).
While deep learning models achieve strong performance on emotion classification, their decision-making processes
remain largely opaque. To bridge the gap between model predictions and neurophysiological interpretation, we
analyze GCN attribution maps and quantitatively align them with emotion-specific EEG microstate dynamics.
Experiments are conducted on the SEED-IV dataset to examine whether GCN models capture meaningful and
physiologically grounded patterns of emotional brain activity.




Our code utilizes different public repos on GitHub:

1. `datasets.py` and `loader.py` refer the public repo *EEGain*: https://github.com/EmotionLab/EEGain
2. The model part refers the benchmark work *LibEER*: https://github.com/XJTU-EEG/LibEER/tree/main


## Repository Structure

```text
gcn-emotion/
├── data/           # Dataset definitions, dataloaders, and preprocessing
├── experiments/    # Training and evaluation scripts for different experiments
├── models/         # GNN models 
├── utils/          # Utility functions (seed, metrics, etc.)
├── main.py         # Entry point for running experiments
├── run.slurm       # Slurm script for cluster execution
├── requirements.txt
└── README.md
```

- `data/`  
  Contains dataset classes, data loading logic, and preprocessing utilities.

- `models/`  
  Model definitions. Each model takes node features as input and outputs class logits.

- `experiments/`  
  Experiment scripts. Each file corresponds to one experimental setting
  (e.g., standard training, mismatch experiments).

- `utils/`  
  Shared utilities such as random seed control and evaluation helpers.

- `main.py`  
  The main entry point. It orchestrates experiments by calling functions in `experiments/`.

## Minimal Pipeline

Some functions and classes we implemented were not used finally. So here is a **minimal pipeline** showing the minimal set of functions and classes that are actually used in the current experiment.


