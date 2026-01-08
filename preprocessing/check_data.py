import os
import glob
import numpy as np
import scipy.io

# ==========================================
# Configuration
# ==========================================
# 你生成的路径
MY_DATA_ROOT = "/public/home/hugf2022/emotion/seediv/m_eeg_feature_smooth_norm"
# 官方数据的路径
OFFICIAL_DATA_ROOT = "/public/home/hugf2022/emotion/seediv/eeg_feature_smooth"

def get_directory_size(path):
    """计算目录总大小 (MB)"""
    total_size = 0
    file_count = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp): # 排除损坏链接
                total_size += os.path.getsize(fp)
                file_count += 1
    return total_size / (1024 * 1024), file_count # 返回 MB 和 文件数

def compare_file_content(my_file_path, official_file_path):
    """对比单个文件的内容详情"""
    print(f"\n--- Deep Dive: {os.path.basename(my_file_path)} ---")
    
    try:
        my_mat = scipy.io.loadmat(my_file_path)
        off_mat = scipy.io.loadmat(official_file_path)
    except Exception as e:
        print(f"Error loading .mat files: {e}")
        return

    # 获取非系统 Key (排除 __header__ 等)
    my_keys = sorted([k for k in my_mat.keys() if not k.startswith('__')])
    off_keys = sorted([k for k in off_mat.keys() if not k.startswith('__')])

    print(f"Key Count: My Data [{len(my_keys)}] vs Official [{len(off_keys)}]")
    
    # 找一个共同的 Key 进行对比 (比如 de_LDS1)
    common_keys = list(set(my_keys) & set(off_keys))
    if not common_keys:
        print("Warning: No common keys found to compare arrays!")
        return
    
    # 选取第一个共同 Key
    sample_key = common_keys[0]
    my_arr = my_mat[sample_key]
    off_arr = off_mat[sample_key]
    
    print(f"Comparing Variable: '{sample_key}'")
    print(f"  > My Shape:       {my_arr.shape}")
    print(f"  > Official Shape: {off_arr.shape}")
    print(f"  > My Dtype:       {my_arr.dtype}")
    print(f"  > Official Dtype: {off_arr.dtype}")
    
    # 计算形状差异
    if len(my_arr.shape) == len(off_arr.shape) == 3:
        # 假设形状是 (Channel, Time, Band)
        time_diff = off_arr.shape[1] - my_arr.shape[1]
        print(f"  > Time Step Diff: {time_diff} (Official - Mine)")
        if time_diff > 0:
            print(f"    Analysis: Official data has {time_diff} more time steps.")
    
    # 检查是否因为官方文件里有额外变量
    only_in_official = set(off_keys) - set(my_keys)
    if only_in_official:
        print(f"  > Extra Keys in Official: {list(only_in_official)[:5]} ... (Total {len(only_in_official)})")

def main():
    print("Checking Data Sizes...")
    
    # 1. 总体积对比
    my_size_mb, my_count = get_directory_size(MY_DATA_ROOT)
    off_size_mb, off_count = get_directory_size(OFFICIAL_DATA_ROOT)
    
    print(f"\n{'='*40}")
    print(f"Total Comparison:")
    print(f"{'Source':<15} | {'Size (MB)':<15} | {'File Count':<10}")
    print(f"{'-'*45}")
    print(f"{'My Data':<15} | {my_size_mb:.2f} MB       | {my_count}")
    print(f"{'Official':<15} | {off_size_mb:.2f} MB       | {off_count}")
    print(f"{'-'*45}")
    
    if off_size_mb > 0:
        ratio = off_size_mb / my_size_mb if my_size_mb > 0 else 0
        print(f"Official is {ratio:.2f}x larger than My Data.")
    
    # 2. 抽样对比 (找一个 Session 1 的文件)
    # 假设目录结构是 root/1/1_20160518.mat
    print(f"\n{'='*40}")
    print("Inspecting a sample file for structural differences...")
    
    # 尝试寻找一个共同存在的文件
    sample_files = glob.glob(os.path.join(MY_DATA_ROOT, "*", "*.mat"))
    if not sample_files:
        print("No files found in my data folder.")
        return
        
    found_match = False
    for my_f in sample_files:
        # 构造对应的官方文件路径
        # my_f 类似: .../m_eeg_feature_smooth/1/1_20160518.mat
        rel_path = os.path.relpath(my_f, MY_DATA_ROOT)
        off_f = os.path.join(OFFICIAL_DATA_ROOT, rel_path)
        
        if os.path.exists(off_f):
            # 找到匹配的文件，进行对比
            print(f"Found match: {rel_path}")
            print(f"  Size My:      {os.path.getsize(my_f)/1024:.2f} KB")
            print(f"  Size Official:{os.path.getsize(off_f)/1024:.2f} KB")
            
            compare_file_content(my_f, off_f)
            found_match = True
            break
    
    if not found_match:
        print("Could not find a matching file name in both directories to compare.")

if __name__ == "__main__":
    main()