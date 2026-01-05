# Excerpt from the Github repo "EEGain", file "dataset.py". 
# See more details at https://github.com/EmotionLab/EEGain/blob/main/eegain/data/datasets.py

import os
import mne
import torch
import numpy as np
import pandas as pd
import scipy.io
import logging

from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
from .transforms import Construct

logger = logging.getLogger("Dataset")


class SeedIVFeatureDataset(Dataset):
    """
    SEED-IV feature-level dataset for DGCNN.

    Each sample corresponds to:
        - one sliding window (segment)
        - one graph with 62 nodes
        - node feature dimension = 5 (DE bands)

    Returned sample:
        x: (62, 5)
        y: int in {0,1,2,3}
    """

    session_labels = {
        1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
        2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
        3: [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
    }

    def __init__(
        self,
        root: str,
        feature_key: str = "de_LDS",
        sessions: List[int] = [1, 2, 3],
        subjects: List[int] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            root: path to eeg_feature_smooth
            feature_key: 'de_LDS' or 'psd_LDS'
            sessions: which sessions to load
            subjects: optional subject id list
        """
        self.root = root
        self.feature_key = feature_key
        self.sessions = sessions
        self.dtype = dtype

        self.samples: List[Tuple[np.ndarray, int]] = []
        self._build_index(subjects)

    def _build_index(self, subjects: List[int] | None):
        for session_id in self.sessions:
            session_dir = os.path.join(self.root, str(session_id))
            label_list = self.session_labels[session_id]

            for file_name in sorted(os.listdir(session_dir)):
                if not file_name.endswith(".mat"):
                    continue

                subject_id = int(file_name.split("_")[0])
                if subjects is not None and subject_id not in subjects:
                    continue

                mat_path = os.path.join(session_dir, file_name)
                mat_data = scipy.io.loadmat(mat_path)

                for trial_idx in range(24):
                    key = f"{self.feature_key}{trial_idx + 1}"
                    if key not in mat_data:
                        continue

                    trial_feature = mat_data[key]  # (62, T, 5)
                    label = label_list[trial_idx]

                    # Split into segments
                    for t in range(trial_feature.shape[1]):
                        segment = trial_feature[:, t, :]  # (62, 5)
                        self.samples.append((segment, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y = self.samples[index]
        x = torch.tensor(x, dtype=self.dtype)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


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


