# Run: srun -p bme_cpu --cpus-per-task=2 --mem=8G python preprocessing/check_data.py


import os
import random
import glob
import numpy as np
import scipy.io

# ==========================================
# Configuration
# ==========================================
# 你的输出目录
OUTPUT_ROOT = "/public/home/hugf2022/emotion/seediv/m_eeg_feature_smooth"
# 想要随机检查的文件数量
NUM_SAMPLES = 5 

def check_single_file(file_path):
    print(f"\n{'='*60}")
    print(f"Checking File: {os.path.basename(file_path)}")
    print(f"Path: {file_path}")
    
    try:
        mat_data = scipy.io.loadmat(file_path)
    except Exception as e:
        print(f"ERROR: Failed to load .mat file. Reason: {e}")
        return

    # 过滤出包含数据的 Key (排除 __header__ 等系统 Key)
    data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
    
    if not data_keys:
        print("WARNING: No data keys found in this file!")
        return

    print(f"Found {len(data_keys)} trials.")
    
    # 遍历每一个 trial 进行检查
    # 为了版面整洁，如果 trial 太多，只打印前3个和异常的
    valid_count = 0
    
    for key in sorted(data_keys):
        data = mat_data[key]
        
        # 1. Check Type
        if not isinstance(data, np.ndarray):
            print(f"  [!] Key '{key}' is not numpy array.")
            continue
            
        # 2. Check Shape
        # 预期形状: (62, Windows, 5)
        shape = data.shape
        is_shape_valid = (len(shape) == 3) and (shape[0] == 62) and (shape[2] == 5)
        
        # 3. Check Values (NaN / Inf)
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        
        status_symbol = "✓" if (is_shape_valid and not has_nan and not has_inf) else "❌"
        
        print(f"  [{status_symbol}] Key: {key:<10} | Shape: {str(shape):<15} | Range: [{data.min():.2f}, {data.max():.2f}]")

        if not is_shape_valid:
            print(f"      >>> ERROR: Shape mismatch! Expected (62, T, 5)")
        if has_nan:
            print(f"      >>> ERROR: Contains NaN values!")
        if has_inf:
            print(f"      >>> ERROR: Contains Inf values!")
            
        if is_shape_valid and not has_nan and not has_inf:
            valid_count += 1

    print(f"Result: {valid_count}/{len(data_keys)} keys passed validation.")

def main():
    if not os.path.exists(OUTPUT_ROOT):
        print(f"Error: Output root {OUTPUT_ROOT} does not exist.")
        return

    # 1. 收集所有生成的 .mat 文件
    # 假设结构是 OUTPUT_ROOT/session_id/xxx.mat
    all_files = glob.glob(os.path.join(OUTPUT_ROOT, "*", "*.mat"))
    
    if not all_files:
        print("Error: No .mat files found in the output directory.")
        return

    total_files = len(all_files)
    print(f"Found total {total_files} processed files.")
    
    # 2. 随机抽样
    sample_size = min(NUM_SAMPLES, total_files)
    selected_files = random.sample(all_files, sample_size)
    
    print(f"Randomly selected {sample_size} files for inspection...")
    
    # 3. 执行检查
    for f_path in selected_files:
        check_single_file(f_path)
        
    print(f"\n{'='*60}")
    print("Check Complete.")

if __name__ == "__main__":
    main()