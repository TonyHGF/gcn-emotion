import os
import numpy as np
import scipy.io
from scipy.signal import butter, filtfilt
import re

# ==========================================
# 1. Configuration / 配置
# ==========================================
INPUT_ROOT = "/public/home/hugf2022/emotion/seediv/eeg_raw_data"
OUTPUT_ROOT = "/public/home/hugf2022/emotion/seediv/m_eeg_feature_smooth_norm" # 修改了输出目录名以区分

FS = 200                # 采样率
WINDOW_SEC = 4          # 窗口时长
WINDOW_SIZE = int(WINDOW_SEC * FS)  # 800 samples

# 频段定义
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 14),
    'beta':  (14, 31),
    'gamma': (31, 50)
}
BAND_ORDER = ['delta', 'theta', 'alpha', 'beta', 'gamma']
SESSIONS = [1, 2, 3]

# ==========================================
# 2. Helper Functions / 辅助函数
# ==========================================

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """5阶巴特沃斯带通滤波"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # filtfilt 实现零相位滤波
    y = filtfilt(b, a, data, axis=-1)
    return y

def compute_de(signal):
    """
    计算微分熵 (DE)
    DE = 0.5 * log(2 * pi * e * variance)
    """
    variance = np.var(signal, axis=-1)
    # 加上微小值避免 log(0)
    variance = np.maximum(variance, 1e-10)
    de = 0.5 * np.log(2 * np.pi * np.e * variance)
    return de

def apply_smoothing(data, window_len=5):
    """
    平滑处理 (LDS Proxy)
    使用移动平均来模拟时序平滑，去除高频抖动
    Args:
        data: (N_channels, T_windows)
    """
    if data.shape[1] < window_len:
        return data
    
    # 简单的移动平均核
    kernel = np.ones(window_len) / window_len
    
    # 沿时间轴 (axis=1) 进行卷积
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode='same'), 
        axis=1, 
        arr=data
    )
    return smoothed

def find_eeg_key(mat_keys, trial_idx):
    """
    动态查找匹配当前 trial 的键名 (后缀匹配)
    解决不同受试者前缀不同 (cz_, wll_, zjy_ 等) 的问题
    """
    suffix = f"eeg{trial_idx}"
    for key in mat_keys:
        if key.startswith('__'): continue
        if key.lower().endswith(suffix):
            return key
    return None

def normalize_features(all_trials_features):
    """
    Session-level Normalization (关键步骤！)
    对该受试者、该 Session 下所有 Trial 的数据进行 Z-score 归一化。
    
    Args:
        all_trials_features: List of (62, T_i, 5) arrays
    Returns:
        normalized_trials: List of (62, T_i, 5) arrays
    """
    if not all_trials_features:
        return []
        
    # 1. 拼接所有时间步以便计算全局均值和方差
    # Concatenate along time dimension (axis 1) -> (62, Total_Time, 5)
    combined_data = np.concatenate(all_trials_features, axis=1)
    
    # 2. 计算均值和标准差 (按通道、按频段独立计算)
    # mean shape: (62, 1, 5)
    mean = np.mean(combined_data, axis=1, keepdims=True)
    std = np.std(combined_data, axis=1, keepdims=True)
    
    # 避免除以零
    std = np.maximum(std, 1e-8)
    
    # 3. 应用归一化到每个 Trial
    normalized_trials = []
    for trial_data in all_trials_features:
        # Broadcasting: (62, T, 5) - (62, 1, 5)
        norm_data = (trial_data - mean) / std
        normalized_trials.append(norm_data)
        
    return normalized_trials

# ==========================================
# 3. Main Processing Logic / 主逻辑
# ==========================================

def process_session(session_id):
    input_dir = os.path.join(INPUT_ROOT, str(session_id))
    output_dir = os.path.join(OUTPUT_ROOT, str(session_id))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Processing Session {session_id}...")
    
    # 获取所有 .mat 文件
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
            
        keys_in_file = list(mat_data.keys())
        
        # 暂存该 Subject 所有 24 个 Trial 的特征，用于后续归一化
        # Dictionary structure: trial_idx -> feature_matrix (62, T, 5)
        temp_features = {} 
        valid_trial_indices = []

        # --- Step 1: 提取特征 ---
        for trial_idx in range(1, 25): 
            raw_key = find_eeg_key(keys_in_file, trial_idx)
            
            if raw_key is None:
                # 某些文件可能缺失部分 trial，跳过
                continue
                
            raw_signal = mat_data[raw_key] # (62, Samples)
            
            # A. 截断 (Truncate)
            n_samples = raw_signal.shape[1]
            n_windows = n_samples // WINDOW_SIZE
            trunc_len = n_windows * WINDOW_SIZE
            
            if n_windows == 0:
                print(f"     Warning: Trial {trial_idx} too short, skipping.")
                continue

            raw_signal = raw_signal[:, :trunc_len]
            features_per_band = []

            # B. 分频段提取特征 (Extract & Smooth)
            for band_name in BAND_ORDER:
                low, high = BANDS[band_name]
                
                # Filter
                filtered_signal = butter_bandpass_filter(raw_signal, low, high, FS)
                
                # Reshape (62, n_windows, 800)
                reshaped_signal = filtered_signal.reshape(62, n_windows, WINDOW_SIZE)
                
                # Compute DE -> (62, n_windows)
                de_features = compute_de(reshaped_signal)
                
                # Smooth -> (62, n_windows)
                de_smoothed = apply_smoothing(de_features)
                
                features_per_band.append(de_smoothed)
            
            # Stack -> (62, n_windows, 5)
            trial_feature_matrix = np.stack(features_per_band, axis=-1)
            
            temp_features[trial_idx] = trial_feature_matrix
            valid_trial_indices.append(trial_idx)
            
        # --- Step 2: 归一化 (Normalization) ---
        if not valid_trial_indices:
            print(f"     Warning: No valid trials found in {file_name}")
            continue

        # 将字典转为列表，保持顺序以便之后还原
        list_of_features = [temp_features[i] for i in valid_trial_indices]
        
        # 执行归一化
        normalized_list = normalize_features(list_of_features)
        
        # --- Step 3: 保存结果 ---
        output_data = {}
        
        # 将归一化后的数据放回字典，Key 保持 de_LDS{i} 格式以便 Dataset 读取
        for idx, norm_feat in zip(valid_trial_indices, normalized_list):
            output_key = f"de_LDS{idx}"
            output_data[output_key] = norm_feat
            
        scipy.io.savemat(save_path, output_data)
        print(f"     Saved {file_name}: {len(output_data)} trials processed & normalized.")

def main():
    if not os.path.exists(INPUT_ROOT):
        print(f"Error: Input path does not exist: {INPUT_ROOT}")
        return

    for sess in SESSIONS:
        process_session(sess)
        
    print("\nProcessing Complete! Check directory:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()