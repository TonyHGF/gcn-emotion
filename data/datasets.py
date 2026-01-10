import os
import torch
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
        feature_keys: Union[str, List[str]] = ["de_LDS"], # Changed to accept list or str
        sessions: List[int] = [1, 2, 3],
        subjects: Optional[List[int]] = None,
        trials: Optional[List[Tuple]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            root: path to eeg_feature_smooth
            feature_keys: Single string or list of strings. 
                          Options: 'de_LDS', 'de_movingAve', 'psd_LDS', 'psd_movingAve'
            sessions: which sessions to load
            subjects: optional subject id list
        """
        self.root = root
        
        # Ensure feature_keys is always a list for consistent processing
        if isinstance(feature_keys, str):
            self.feature_keys = [feature_keys]
        else:
            self.feature_keys = feature_keys
            
        self.sessions = sessions
        self.dtype = dtype
        self.trials = trials

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._build_index(subjects)

    def _build_index(self, subjects: List[int] | None):
        for session_id in self.sessions:
            session_dir = os.path.join(self.root, str(session_id))
            label_list = session_labels[session_id]

            # Check if directory exists
            if not os.path.exists(session_dir):
                logger.warning(f"Session directory not found: {session_dir}")
                continue

            for file_name in sorted(os.listdir(session_dir)):
                if not file_name.endswith(".mat"):
                    continue

                subject_id = int(file_name.split("_")[0])
                if subjects is not None and subject_id not in subjects:
                    continue

                mat_path = os.path.join(session_dir, file_name)
                try:
                    mat_data = scipy.io.loadmat(mat_path)
                except Exception as e:
                    logger.error(f"Error loading {mat_path}: {e}")
                    continue

                for trial_idx in range(24):
                    if self.trials is not None and (session_id, trial_idx) not in self.trials:
                        continue
                    
                    # --- NEW LOGIC START ---
                    collected_features = []
                    valid_trial = True
                    
                    # Iterate through all requested feature types (e.g., de_LDS, psd_LDS)
                    for f_key in self.feature_keys:
                        # Construct key, e.g., "de_LDS1", "psd_LDS1"
                        key = f"{f_key}{trial_idx + 1}"
                        
                        if key not in mat_data:
                            valid_trial = False
                            # logger.warning(f"Key {key} missing in {file_name}")
                            break
                        
                        # Data shape: (62, T, 5)
                        data = mat_data[key]
                        collected_features.append(data)

                    if not valid_trial:
                        continue
                    
                    # Check if time dimensions align (e.g. if de_LDS has T=60 but psd has T=59)
                    # This prevents concatenation errors
                    base_time_len = collected_features[0].shape[1]
                    if any(f.shape[1] != base_time_len for f in collected_features):
                        logger.warning(f"Time dimension mismatch in {file_name} trial {trial_idx}")
                        continue

                    # Concatenate along the last axis (bands)
                    # Shape becomes: (62, T, 5 * num_keys)
                    trial_feature = np.concatenate(collected_features, axis=-1)
                    # --- NEW LOGIC END ---

                    label = label_list[trial_idx]

                    # Split into segments
                    for t in range(trial_feature.shape[1]):
                        segment = trial_feature[:, t, :]  # (62, total_bands)
                        self.samples.append((segment, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y = self.samples[index]
        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(y, dtype=torch.long)
        return x, y