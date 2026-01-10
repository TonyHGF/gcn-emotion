import os
import torch
import random
import numpy as np
import scipy.io
import logging
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Union

# Set up logger
logger = logging.getLogger("Dataset")

# 6 per session
session_labels = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
}

class SeedIVFeatureDataset(Dataset):
    """
    SEED-IV feature-level dataset for DGCNN with flexible feature selection.

    Each sample corresponds to:
        - one sliding window (segment)
        - one graph with 62 nodes
        - node feature dimension = 5 * len(feature_keys)
          (e.g., if using ['de_LDS', 'psd_LDS'], dim = 10)

    Returned sample:
        x: (62, 5 * N)
        y: int in {0,1,2,3}
    """

    def __init__(
        self,
        root: str,
        feature_keys: Union[str, List[str]] = ["de_LDS"],
        sessions: List[int] = [1, 2, 3],
        split: str = "train",                  # train / val / test
        subjects: Optional[List[int]] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
    ):
        self.root = root
        self.sessions = sessions
        self.split = split
        self.subjects = subjects
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.dtype = dtype

        if isinstance(feature_keys, str):
            self.feature_keys = [feature_keys]
        else:
            self.feature_keys = feature_keys

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._build_index()


    def _build_index(self):
        # Make train/val/test deterministic & consistent across three datasets
        base_seed = self.seed

        for subject_id in self._iter_subject_ids():
            if self.subjects is not None and subject_id not in self.subjects:
                continue

            # ---- Load this subject's mat for each session once ----
            mat_by_session = {}
            for session_id in self.sessions:
                session_dir = os.path.join(self.root, str(session_id))
                if not os.path.exists(session_dir):
                    continue

                # Assumption: file name starts with subject_id, like "1_xxx.mat"
                target_file = None
                for file_name in sorted(os.listdir(session_dir)):
                    if not file_name.endswith(".mat"):
                        continue
                    if int(file_name.split("_")[0]) == subject_id:
                        target_file = file_name
                        break

                if target_file is None:
                    continue

                mat_path = os.path.join(session_dir, target_file)
                mat_by_session[session_id] = scipy.io.loadmat(mat_path)

            if len(mat_by_session) == 0:
                continue

            # ---- Build 18 trials per label across sessions: (session_id, trial_idx) ----
            trials_by_label = {0: [], 1: [], 2: [], 3: []}
            for session_id in self.sessions:
                if session_id not in mat_by_session:
                    continue
                label_list = session_labels[session_id]  # length 24
                for trial_idx in range(24):
                    lbl = label_list[trial_idx]
                    trials_by_label[lbl].append((session_id, trial_idx))

            # Sanity (SEED-IV typical): 3 sessions × 6 trials per label = 18
            for lbl in [0, 1, 2, 3]:
                if len(trials_by_label[lbl]) != 18:
                    # If your files are incomplete, relax/remove this assert
                    raise ValueError(f"Subject {subject_id}: label {lbl} has {len(trials_by_label[lbl])} trials, expected 18.")

            # ---- Subject-local shuffle, then 14/2/2 split for each label ----
            rng = random.Random(base_seed + subject_id * 10007)

            selected_trials = []
            for lbl in [0, 1, 2, 3]:
                trials = trials_by_label[lbl]
                rng.shuffle(trials)

                train_part = trials[:14]
                val_part = trials[14:16]
                test_part = trials[16:18]

                if self.split == "train":
                    selected_trials.extend(train_part)
                elif self.split == "val":
                    selected_trials.extend(val_part)
                else:  # "test"
                    selected_trials.extend(test_part)

            # ---- Read selected trials and append segments ----
            for session_id, trial_idx in selected_trials:
                mat_data = mat_by_session[session_id]
                label = session_labels[session_id][trial_idx]

                collected = []
                valid = True
                for fkey in self.feature_keys:
                    key = f"{fkey}{trial_idx + 1}"
                    if key not in mat_data:
                        valid = False
                        break
                    collected.append(mat_data[key])  # (62, T, bands)

                if not valid:
                    continue

                T = collected[0].shape[1]
                if any(arr.shape[1] != T for arr in collected):
                    continue

                trial_feature = np.concatenate(collected, axis=-1)  # (62, T, total_bands)

                for t in range(T):
                    self.samples.append((trial_feature[:, t, :], label))


    def _iter_subject_ids(self):
        # Discover subjects from the first available session folder
        for session_id in self.sessions:
            session_dir = os.path.join(self.root, str(session_id))
            if not os.path.exists(session_dir):
                continue
            subject_ids = []
            for file_name in sorted(os.listdir(session_dir)):
                if file_name.endswith(".mat"):
                    subject_ids.append(int(file_name.split("_")[0]))
            return sorted(set(subject_ids))
        return []


    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y = self.samples[index]
        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(y, dtype=torch.long)
        return x, y