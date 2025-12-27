import torch
import numpy as np
import logging
import sys

from datasets import EEGDataset, SeedIV
from loader import EEGDataloader


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s',
        stream=sys.stdout
    )

    seediv_path = r"/public/home/hugf2022/emotion/seediv/eeg_raw_data"

    dataset = SeedIV(root=seediv_path, label_type="V", ground_truth_threshold=0.5)
    print("Dataset Initialized!")

    subject_ids = dataset.__get_subject_ids__()
    print(f"Subject IDs: {subject_ids}")

    loader = EEGDataloader(dataset, batch_size=4)

    print("Test LOSO:")
    for batch in loader.loso(subject_out_num=1):
        train_data = batch['train']
        for images, labels in train_data:
            print(f"Data load successfully! Shape: {images.shape}")
            break
        break



if __name__ == '__main__':
    main()