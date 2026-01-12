import os
import mne
import torch
import random
import random
import numpy as np
import pandas as pd
import scipy.io
import logging
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Tuple, Optional, Union
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
from .transforms import Construct

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

def within_subject_split(config):
    """
    Within-Subject Experiment Split:
    - Subject: Single subject specified by config['target_subject']
    - Train Pool: First 18 trials of each session (Indices 0-17)
    - Test Pool: Last 6 trials of each session (Indices 18-23)
    - Validation: Randomly split from Train Pool (e.g., 20%)
    """
    target_subject = config.get("target_subject")
    if target_subject is None:
        raise ValueError("config['target_subject'] must be provided for within-subject split.")

    # 1. 定义 Trial 的 index 范围
    # SEED-IV 约定：前18个做训练，后6个做测试
    train_pool_indices = list(range(0, 18))  # 0 to 17
    test_pool_indices = list(range(18, 24))  # 18 to 23
    
    # 2. 生成 (Session, Trial_Idx) 列表
    # 这里的 train_pool_candidates 包含 3 * 18 = 54 个 trial
    train_pool_candidates = []
    test_trials = []
    
    for sess in [1, 2, 3]:
        for t_idx in train_pool_indices:
            train_pool_candidates.append((sess, t_idx))
        for t_idx in test_pool_indices:
            test_trials.append((sess, t_idx))
            
    # 3. 从 Train Pool 中划分 Validation Set
    # 为了保证训练稳定性，我们设置随机种子进行打乱
    # 注意：这里的 seed 应该固定，保证同一个人在不同超参实验中数据集是一样的
    rng = random.Random(config.get("seed", 42))
    rng.shuffle(train_pool_candidates)
    
    # 计算验证集大小 (例如 20% 的训练池)
    val_ratio = config.get("val_ratio", 0.2)
    num_val = int(len(train_pool_candidates) * val_ratio)
    
    val_trials = train_pool_candidates[:num_val]
    train_trials = train_pool_candidates[num_val:]
    
    # 4. 创建 Dataset
    # 关键点：subjects=[target_subject]，保证只加载这一个人的数据
    # 然后用 allowed_trials (即 Dataset 中的 trials 参数) 来过滤具体的 trial
    
    # 训练集
    train_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key="de_LDS",
        sessions=[1, 2, 3],
        subjects=[target_subject], # 只加载当前被试
        trials=train_trials        # 只加载前18个中分出来的训练部分
    )
    
    # 验证集
    val_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key="de_LDS",
        sessions=[1, 2, 3],
        subjects=[target_subject],
        trials=val_trials
    )
    
    # 测试集 (固定的后6个)
    test_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key="de_LDS",
        sessions=[1, 2, 3],
        subjects=[target_subject],
        trials=test_trials
    )
    
    # 5. Loaders
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_set, batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader, len(train_set), len(val_set), len(test_set)

def all_mix_split(config):
    # ---------- Dataset ----------
    dataset = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key="de_LDS",
        sessions=[1, 2, 3],)

    num_total = len(dataset)
    num_train = int(num_total * config["train_ratio"])
    num_val = int(num_total * config["val_ratio"])
    num_test = num_total - num_train - num_val

    train_set, val_set, test_set = random_split(
        dataset, [num_train, num_val, num_test])

    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader, num_train, num_val, num_test

# Divide based on subject
def loso_split(config):
    pick = random.randint(1, 15)
    
    # Test: 1
    test_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key="de_LDS",
        sessions=[1, 2, 3],
        subjects=[pick])
    #Train: 14
    train_set = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key="de_LDS",
        sessions=[1, 2, 3],
        subjects=[i for i in range(1, 16) if (i != pick)])
    
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False)

    return train_loader, None, test_loader, len(train_set), 0, len(test_set)

def trial_split(config):

    def generate_balanced_trial_splits():
        trials_by_label = {0: [], 1: [], 2: [], 3: []}
        
        for sess in [1, 2, 3]:
            labels = session_labels[sess]
            for idx, lbl in enumerate(labels):
                trials_by_label[lbl].append((sess, idx))
        
        train_list, val_list, test_list = [], [], []

        for lbl in [0, 1, 2, 3]:
            trials = trials_by_label[lbl]
            assert len(trials) == 18, f"Expected 18 trials for label {lbl}, got {len(trials)}"
            random.shuffle(trials)

            train_chunk = trials[:14]
            val_chunk = trials[14:16]
            test_chunk = trials[16:]

            train_list.extend(train_chunk)
            val_list.extend(val_chunk)
            test_list.extend(test_chunk)
        
        return train_list, val_list, test_list
    
    train_trials, val_trials, test_trials = generate_balanced_trial_splits()

    train_set = SeedIVFeatureDataset(
        root=config["data_root"], feature_key="de_LDS", sessions=[1, 2, 3],
        trials=train_trials)
    val_set = SeedIVFeatureDataset(
        root=config["data_root"], feature_key="de_LDS", sessions=[1, 2, 3],
        trials=val_trials)
    test_set = SeedIVFeatureDataset(
        root=config["data_root"], feature_key="de_LDS", sessions=[1, 2, 3],
        trials=test_trials)
    
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(
        val_set, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(
        test_set, batch_size=config["batch_size"], shuffle=False)
    
    return train_loader, val_loader, test_loader, len(train_set), len(val_set), len(test_set)

class EEGDatasetBase(Dataset):
    """
    Python dataset wrapper that takes tensors and implements dataset structure
    """
    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        self.x = x_tensor
        self.y = y_tensor

        assert len(self.x) == len(self.y)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]
    
    def __len__(self) -> int:
        return len(self.y)
    

class EEGDataset(ABC):
    @abstractmethod
    def __get_subject_ids__(self) -> List[int]:
        raise NotImplementedError

    @abstractmethod
    def __get_subject__(
        self, subject_index: int
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, int]]:
        raise NotImplementedError
    

class SeedIV(EEGDataset):
    @staticmethod
    def _create_user_recording_mapping(data_path: Path, unique_identifier: str) -> Dict[int, List[str]]:
        """This method creates mapping between users and user_recordings
        Args:
            data_path(Path): path to SEED IV dataset

        Returns:
            user_session_info(Dict[int, List[str]]): This is the dictionary where key is user_id and value is list
                                                     of file_names that's associated to this particular user
        """

        num_sessions = 3  # There are three sessions in SEED IV dataset
        num_trials = 24
        session = 1 # we decided to test just the first session 
        user_session_info: Dict[int, List[str]] = defaultdict(list)

        # for session in range(1, num_sessions + 1):
        path = (
            data_path / Path(str(session)) # eeg_raw_data
        )  # Path to particular sessions mat files
        file_paths = os.listdir(path)
        print("Subject File Names:")
        for mat_file_name in file_paths:
            if "label" not in mat_file_name and "channel" not in mat_file_name and ".mat" in mat_file_name:
                subject_id = int(
                    mat_file_name[: mat_file_name.index("_")]
                )
                subject_file_name = str(path) + "/" + mat_file_name
                print(subject_file_name)
                 # file name starts with user_id
                # user_session_info[subject_id].append(
                #     str(session) + "/" + subject_file_name
                # )
                session_file_name = subject_file_name
                for i in range(num_trials):
                    curr_session_trial_name = str(i) + unique_identifier + session_file_name
                    user_session_info[subject_id].append(curr_session_trial_name)            

        # logger.debug(f"Subject id -> sessions: {user_session_info}")
        return user_session_info

    def read_mat_file(mapping_list, unique_identifier):
        user_ids = mapping_list.keys()
        already_read_sessions = {}
        ids_and_read_data = {}
        for user in user_ids:
            trial_names = mapping_list[user] # "0_../Desktop/seed_data_cp/Preprocessed_EEG/1_20131027.mat"
            for trial_name in trial_names:
                session_name = "".join(trial_name.split(unique_identifier)[1])
                trial_id = int("".join(trial_name.split(unique_identifier)[0][:len(unique_identifier)]))
                if session_name in already_read_sessions:
                    session = already_read_sessions[session_name]
                    eg_data = session[trial_id]
                    ids_and_read_data[trial_name] = eg_data
                else:
                    path_to_mat = Path(str(session_name))
                    mat_data = scipy.io.loadmat(path_to_mat)  # Get Matlab File
                    mat_data_values = list(mat_data.values())[
                                3:] 
                    already_read_sessions[session_name] = mat_data_values
                    eg_data = mat_data_values[trial_id]
                    ids_and_read_data[trial_name] = eg_data
        return ids_and_read_data 

    def __init__(self, root: str, label_type: str, ground_truth_threshold, transform: Construct = None, **kwargs):
        """This is just constructor for SEED IV class
        Args:
            root(str): Path to SEED IV dataset folder
            label_type(str): 'V' for Valence and 'A' for Arousal
            transform(Construct): transformations to apply on dataset

        Return:
        """

        self.root = Path(root)
        self.ground_truth_threshold = ground_truth_threshold
        self.transform = transform
        self.unique_identifier = "<EOF>"
        self.mapping_list = SeedIV._create_user_recording_mapping(self.root, self.unique_identifier)
        self.read_files = Seed.read_mat_file(self.mapping_list, self.unique_identifier)
        self.label_type = label_type
        self._num_trials = 24
        self._sampling_rate = 1000

        # logger.info(f"Using Dataset: {self.__class__.__name__}")
        # logger.info(f"Using label: {self.label_type}")

    def __get_subject_ids__(self) -> List[int]:
        """Method returns list of subject ids"""
        return list(self.mapping_list.keys())

    def __get_subject__(
        self, subject_index: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """This method returns data and associated labels for specific subject

        Args:
            subject_index(int): user id

        Returns:
            data_array(Dict[str, np.ndarray]): Dictionary of files and data associated to specific user
            label_array(Dict[str, int]): labels for each recording
        """

        labels_file_path = self.root / Path("ReadMe.txt")
        path_to_channels_excel = self.root / Path("Channel Order.xlsx")
        channels_file = pd.read_excel(path_to_channels_excel, header=None)
        channels = list(channels_file.iloc[:, 0])

        num_trials = self._num_trials
        sampling_rate = self._sampling_rate
        sessions = self.mapping_list[subject_index]
        data_array, label_array = {}, {}

        for session in sessions:
            session_name = "".join(session.split(self.unique_identifier)[1])
            trial_id = int("".join(session.split(self.unique_identifier)[0][:len(self.unique_identifier)]))            
            eeg_data = self.read_files[session] # each matlab file contains 15 trials. Here we take one trial
            info = mne.create_info(
                ch_names=channels, sfreq=sampling_rate, ch_types="eeg"
            )
            raw_data = mne.io.RawArray(
                eeg_data, info, verbose=False
            )  # convert numpy ndarray to mne object

            if self.transform:  # apply transformations
                raw_data = self.transform(raw_data)

            session_trial = session + "/" + str(trial_id)
            data_array[session_trial] = raw_data.get_data()

            with open(labels_file_path, "r") as file:
                labels_file_content = file.read()

            # Extract label from file and add to label_array
            session_id = session[: session.index("/")]
            # since we are using just first session :  session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
            session_id = 1
            session_start_idx = labels_file_content.index(
                f"session{session_id}_label"
            )
            session_end_idx = labels_file_content.index(";", session_start_idx)
            session_labels = labels_file_content[session_start_idx:session_end_idx]
            session_labels = eval(session_labels[session_labels.index("[") :])

            emotional_label = session_labels[trial_id]
            label_array[session_trial] = emotional_label

        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        # logger.debug(
        #     f"subj index: {subject_index} data {data_array.keys()}, label {label_array.keys()}"
        # )

        return data_array, label_array


    def __get_trials__(self, sessions, subject_ids):
        labels_file_path = self.root / Path("ReadMe.txt")
        path_to_channels_excel = self.root / Path("Channel Order.xlsx")
        channels_file = pd.read_excel(path_to_channels_excel, header=None)
        channels = list(channels_file.iloc[:, 0])

        num_trials = self._num_trials
        sampling_rate = self._sampling_rate
        data_array, label_array = {}, {}

        for session in sessions:
            session_name = "".join(session.split(self.unique_identifier)[1])
            trial_id = int("".join(session.split(self.unique_identifier)[0][:len(self.unique_identifier)]))            
            eeg_data = self.read_files[session] # each matlab file contains 24 trials. Here we take one trial
            info = mne.create_info(
                ch_names=channels, sfreq=sampling_rate, ch_types="eeg"
            )
            raw_data = mne.io.RawArray(
                eeg_data, info, verbose=False
            )  # convert numpy ndarray to mne object

            if self.transform:  # apply transformations
                raw_data = self.transform(raw_data)

            session_trial = session_name + "/" + str(trial_id)
            data_array[session_trial] = raw_data.get_data()

            with open(labels_file_path, "r") as file:
                labels_file_content = file.read()

            # Extract label from file and add to label_array
            session_id = session[: session.index("/")]
            # since we are using just first session :  session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
            session_id = 1
            session_start_idx = labels_file_content.index(
                f"session{session_id}_label"
            )
            session_end_idx = labels_file_content.index(";", session_start_idx)
            session_labels = labels_file_content[session_start_idx:session_end_idx]
            session_labels = eval(session_labels[session_labels.index("["):])

            emotional_label = session_labels[trial_id]
            label_array[session_trial] = emotional_label

        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array
            

class Seed(EEGDataset):
    @staticmethod
    def _create_user_mat_mapping(data_path: Path, unique_identifier: str) -> Dict[int, List[str]]:
        """This method creates mapping between users and user_recordings
        Args:
            data_path(Path): path to SEED dataset

        Returns:
            user_session_info(Dict[int, List[str]]): This is the dictionary where key is user_id and value is list
                                                     of file_names that's associated to this particular user
        """

        # num_sessions = 3  # There are three sessions in SEED IV dataset
        user_session_info: Dict[int, List[str]] = defaultdict(list)
        path = (
                data_path  # eeg_preprocessed_data
        )
        file_paths = os.listdir(path)
        for mat_file_name in file_paths:
            if "label" not in mat_file_name and "channel" not in mat_file_name and ".mat" in mat_file_name:
                subject_id = int(
                    mat_file_name[: mat_file_name.index("_")]
                )  # file name starts with user_id
                session_file_name = str(path) + "/" + mat_file_name
                for i in range(15):
                    curr_session_trial_name = str(i) + unique_identifier + session_file_name
                    user_session_info[subject_id].append(curr_session_trial_name)

        return user_session_info
    
    def read_mat_file(mapping_list, unique_identifier):
        user_ids = mapping_list.keys()
        already_read_sessions = {}
        ids_and_read_data = {}
        for user in user_ids:
            trial_names = mapping_list[user] # "0_../Desktop/seed_data_cp/Preprocessed_EEG/1_20131027.mat"
            for trial_name in trial_names:
                session_name = "".join(trial_name.split(unique_identifier)[1])
                trial_id = int("".join(trial_name.split(unique_identifier)[0][:len(unique_identifier)]))
                if session_name in already_read_sessions:
                    session = already_read_sessions[session_name]
                    eg_data = session[trial_id]
                    ids_and_read_data[trial_name] = eg_data
                else:
                    path_to_mat = Path(str(session_name))
                    mat_data = scipy.io.loadmat(path_to_mat)  # Get Matlab File
                    mat_data_values = list(mat_data.values())[
                                3:] 
                    already_read_sessions[session_name] = mat_data_values
                    eg_data = mat_data_values[trial_id]
                    ids_and_read_data[trial_name] = eg_data
        return ids_and_read_data 



    def __init__(self, root: str, label_type: str, transform: Construct = None, **kwargs):
        """This is just constructor for SEED class
        Args:
            root(str): Path to SEED dataset folder
            transform(Construct): transformations to apply on dataset

        Return:
        """

        self.root = Path(root)
        self.transform = transform
        self.unique_identifier = "<EOF>"
        self.mapping_list = Seed._create_user_mat_mapping(self.root, self.unique_identifier)
        self.read_files = Seed.read_mat_file(self.mapping_list, self.unique_identifier)
        self.label_type = label_type
        self._num_trials = 15
        self._sampling_rate = 1000

    def __get_subject_ids__(self) -> List[int]:
        """Method returns list of subject ids"""

        return list(self.mapping_list.keys())

    def __get_subject__(
        self, subject_index: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
        """This method returns data and associated labels for specific subject

        Args:
            subject_index(int): user id

        Returns:
            data_array(Dict[str, np.ndarray]): Dictionary of files and data associated to specific user
            label_array(Dict[str, int]): labels for each recording
        """

        path_to_channels_excel = self.root / Path("channel-order.xlsx")
        channels_file = pd.read_excel(path_to_channels_excel, header=None)
        channels = list(channels_file.iloc[:, 0])

        num_trials = self._num_trials
        sampling_rate = self._sampling_rate
        sessions = self.mapping_list[subject_index]
        data_array, label_array = {}, {}

        for session in sessions:
            session_name = "".join(session.split(self.unique_identifier)[1])
            trial_id = int("".join(session.split(self.unique_identifier)[0][:len(self.unique_identifier)]))            
            eeg_data = self.read_files[session] # each matlab file contains 15 trials. Here we take one trial
            info = mne.create_info(
                ch_names=channels, sfreq=sampling_rate, ch_types="eeg"
            )
            raw_data = mne.io.RawArray(
                eeg_data, info, verbose=False
            )  # convert numpy ndarray to mne object

            if self.transform:  # apply transformations
                raw_data = self.transform(raw_data)

            session_trial = session + "/" + str(trial_id)
            data_array[session_trial] = raw_data.get_data()

            # Extract label from file and add to label_array
            session_id = session[: session.index("/")]
            session_labels = [ 2,  1, 0, 0,  1,  2, 0,  1,  2,  2,  1, 0,  1,  2, 0]

            emotional_label = session_labels[trial_id]
            label_array[session_trial] = emotional_label

        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array

    def __get_trials__(self, sessions, subject_ids):
        path_to_channels_excel = self.root / Path("channel-order.xlsx")
        channels_file = pd.read_excel(path_to_channels_excel, header=None)
        channels = list(channels_file.iloc[:, 0])

        num_trials = self._num_trials
        sampling_rate = self._sampling_rate
        data_array, label_array = {}, {}

        for session in sessions:
            session_name = "".join(session.split(self.unique_identifier)[1])
            trial_id = int("".join(session.split(self.unique_identifier)[0][:len(self.unique_identifier)]))            
            eeg_data = self.read_files[session] # each matlab file contains 15 trials. Here we take one trial
            info = mne.create_info(
                ch_names=channels, sfreq=sampling_rate, ch_types="eeg"
            )
            raw_data = mne.io.RawArray(
                eeg_data, info, verbose=False
            )  # convert numpy ndarray to mne object

            if self.transform:  # apply transformations
                raw_data = self.transform(raw_data)

            session_trial = session_name + "/" + str(trial_id)
            data_array[session_trial] = raw_data.get_data()
            # session_labels = [ 1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1]
            session_labels = [ 2,  1, 0, 0,  1,  2, 0,  1,  2,  2,  1, 0,  1,  2, 0]
            emotional_label = session_labels[trial_id]
            label_array[session_trial] = emotional_label

        data_array = {
            key: np.expand_dims(value, axis=-3) for key, value in data_array.items()
        }

        return data_array, label_array


