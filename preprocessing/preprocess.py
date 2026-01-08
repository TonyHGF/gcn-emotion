import os
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt
import re  # 引入正则模块

# ==========================================
# Configuration (保持不变)
# ==========================================
INPUT_ROOT = "/public/home/hugf2022/emotion/seediv/eeg_raw_data"
OUTPUT_ROOT = "/public/home/hugf2022/emotion/seediv/m_eeg_feature_smooth"

FS = 200 
WINDOW_SEC = 4 
WINDOW_SIZE = int(WINDOW_SEC * FS) 

BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta':  (14, 31),
    'gamma': (31, 50)
}
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']
SESSIONS = [1, 2, 3]

# ... (butter_bandpass_filter, compute_de, apply_smoothing 保持不变) ...
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=-1)
    return y

def compute_de(signal):
    variance = np.var(signal, axis=-1)
    variance = np.maximum(variance, 1e-10)
    de = 0.5 * np.log(2 * np.pi * np.e * variance)
    return de

def apply_smoothing(data, window_len=5):
    if data.shape[1] < window_len:
        return data
    kernel = np.ones(window_len) / window_len
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'), 
        axis=1, 
        arr=data
    )
    return smoothed

def find_eeg_key(mat_keys, trial_idx):
    """
    动态查找匹配当前 trial 的键名。
    例如：trial_idx=1，可能匹配 'cz_eeg1', 'zjy_eeg1', 'eeg1' 等。
    只要是以 'eeg1' 结尾（忽略大小写）且不包含其他无关字符即可。
    """
    # 构造后缀，例如 "eeg1"
    suffix = f"eeg{trial_idx}"
    
    for key in mat_keys:
        # 忽略系统自带的 key (__, __header__ 等)
        if key.startswith('__'):
            continue
            
        # 检查是否以 eegX 结尾 (大小写不敏感)
        if key.lower().endswith(suffix):
            return key
    return None

def process_session(session_id):
    input_dir = os.path.join(INPUT_ROOT, str(session_id))
    output_dir = os.path.join(OUTPUT_ROOT, str(session_id))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Processing Session {session_id}...")
    
    files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.mat')]
    
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        save_path = os.path.join(output_dir, file_name)
        
        print(f"  -> Loading {file_name}...")
        try:
            mat_data = scipy.io.loadmat(file_path)
        except Exception as e:
            print(f"     Error loading {file_name}: {e}")
            continue
            
        output_data = {}
        keys_in_file = list(mat_data.keys())
        
        # Process each of the 24 trials
        for trial_idx in range(1, 25): # 1 to 24
            
            # === 修改点：动态查找 Key ===
            raw_key = find_eeg_key(keys_in_file, trial_idx)
            
            if raw_key is None:
                print(f"     Warning: Could not find key for trial {trial_idx} in {file_name}")
                continue
            # ==========================
                
            raw_signal = mat_data[raw_key]
            
            # 1. Truncate
            n_samples = raw_signal.shape[1]
            n_windows = n_samples // WINDOW_SIZE
            trunc_len = n_windows * WINDOW_SIZE
            
            if n_windows == 0:
                print(f"     Warning: Trial {trial_idx} ({raw_key}) too short.")
                continue

            raw_signal = raw_signal[:, :trunc_len]
            features_per_band = []

            # 2. Extract Features
            for band_name in BAND_ORDER:
                low, high = BANDS[band_name]
                filtered_signal = butter_bandpass_filter(raw_signal, low, high, FS)
                reshaped_signal = filtered_signal.reshape(62, n_windows, WINDOW_SIZE)
                de_features = compute_de(reshaped_signal)
                de_smoothed = apply_smoothing(de_features)
                features_per_band.append(de_smoothed)
            
            trial_feature_matrix = np.stack(features_per_band, axis=-1)
            
            output_key = f"de_LDS{trial_idx}"
            output_data[output_key] = trial_feature_matrix
            
        # Save new .mat file
        scipy.io.savemat(save_path, output_data)
        # 打印一下包含的 Trial 数量，方便确认
        print(f"     Saved {file_name}: {len(output_data)} trials processed.")

def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input path does not exist: {INPUT_ROOT}")
        return

    for sess in SESSIONS:
        process_session(sess)
        
    print("\nProcessing Complete!")

if __name__ == "__main__":
    main()