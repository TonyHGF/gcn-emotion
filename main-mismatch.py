import numpy as np
import logging
import sys
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from dataclasses import dataclass
import matplotlib.pyplot as plt

from data import SeedIV, EEGDatasetBase, EEGDataloader
from models import DGCNN, DGCNNAdapter, Construct, Crop, Segment, Resample
from utils import FeatureExtractorConfig, helpers



def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s',
        stream=sys.stdout
    )

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    seediv_path = r"/public/home/hugf2022/emotion/seediv/eeg_raw_data"

    WINDOW = 4.0 # seconds
    OVERLAP = 2.0 # seconds

    transform = Construct([
        Crop(t_min=1.0, t_max=-1.0),
        Resample(sampling_r=200), 
        Segment(duration=WINDOW, overlap=OVERLAP)
    ])

    # ============ Dataset DEBUG ======================
    # dataset = SeedIV(root=seediv_path, label_type="V", ground_truth_threshold=0.5, transform=transform)
    # print("Dataset Initialized!")

    # subject_ids = dataset.__get_subject_ids__()
    # print(f"Subject IDs: {subject_ids}")

    # helpers.inspect_one_subject(dataset, subject_ids[0])

    # total_segments = helpers.count_total_segments(dataset)
    # print(f"Total segments in dataset: {total_segments}")

    # loader = EEGDataloader(dataset, batch_size=4) # Dimension: (Batch=4, 1, Channel=62, Time=800)
    # ============ End of Dataset DEBUG ===============

    dataset = SeedIV(root=seediv_path, label_type="V", ground_truth_threshold=0.5, transform=transform)
    print("Dataset Initialized!")

    '''
    There are two training strategies: 
    1. Cross-subject: leave some subject out, and train on others. 
       There is an API called "LOSO (Leave One Subject Out)" in the SeedIV dataset class.
       This strategy makes more sense but may have lower accuracy rate.
    2. Put all segments together, then split them randomly. This may have higher accuracy and be easier to write.
       So in the first version, I train and test this model with this strategy.
    '''

    # ============= Split at segment level ================
    all_x = []
    all_y = []

    for subject_id in dataset.__get_subject_ids__():
        data_dict, label_dict = dataset.__get_subject__(subject_id)

        for key in data_dict.keys():
            x = data_dict[key]   # (n_seg, 1, 62, T)
            y = label_dict[key]

            all_x.append(x)
            all_y.append(
                np.repeat(y, x.shape[0])
            )

    all_x = torch.from_numpy(np.concatenate(all_x, axis=0)).float()
    all_y = torch.from_numpy(np.concatenate(all_y, axis=0)).long()

    # Normalization:
    all_x, _ = EEGDataloader.normalize(all_x, all_x.clone())

    num_samples = all_x.shape[0]
    indices = torch.randperm(num_samples)

    train_end = int(0.7 * num_samples)
    val_end   = int(0.85 * num_samples)

    train_idx = indices[:train_end]
    val_idx   = indices[train_end:val_end]
    test_idx  = indices[val_end:]

    train_loader = DataLoader(EEGDatasetBase(all_x[train_idx], all_y[train_idx]), batch_size=32, shuffle=True)
    val_loader = DataLoader(EEGDatasetBase(all_x[val_idx], all_y[val_idx]), batch_size=32, shuffle=False)
    test_loader = DataLoader(EEGDatasetBase(all_x[test_idx], all_y[test_idx]), batch_size=32,shuffle=False)

    logging.info(f"Data Loaded: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")
    # =====================================================


    # Model training (DGCNN):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    model = DGCNNAdapter(
        num_electrodes=62,
        num_classes=4,   # SEED-IV: 4 classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    best_val_acc = 0.0

    # History for plotting
    history = {
        'train_loss': [],
        'val_acc': []
    }

    # Training Phase:
    logging.info("Starting training...")
    for epoch in range(num_epochs):
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
        history['train_loss'].append(avg_loss)

        # --- Validation ---
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
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

        # --- Checkpoint: Save Best Model ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"--> Best model saved at epoch {epoch+1} with Acc: {best_val_acc:.4f}")

    # Test Phase:
    logging.info("Starting Testing Phase...")
    
    # Load the best model weights
    best_model_path = 'checkpoints/best_model.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        logging.info("Loaded best model from checkpoints.")
    else:
        logging.warning("Best model checkpoint not found, using last epoch weights.")

    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            test_correct += (preds == y).sum().item()
            test_total += y.size(0)

    test_acc = test_correct / test_total
    logging.info(f"Final Test Accuracy: {test_acc:.4f}")

    # 7. Generate Diagrams
    logging.info("Generating training curves...")
    
    plt.figure(figsize=(12, 5))

    # Subplot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='blue')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Acc', color='green')
    plt.axhline(y=test_acc, color='red', linestyle='--', label=f'Final Test Acc ({test_acc:.2f})')
    plt.title('Validation Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results/training_curves.png')
    logging.info("Diagram saved to results/training_curves.png")    


if __name__ == '__main__':
    main()