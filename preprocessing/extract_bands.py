import os
import numpy as np
import scipy.io

# ==========================================
# 1. Configuration (用户配置区)
# ==========================================

# [输入] 原始包含5个频段的数据路径
SOURCE_ROOT = "/public/home/hugf2022/emotion/seediv/m_eeg_feature_smooth" 

# [输出] 提取后数据的保存路径 (建议根据提取的频段修改文件夹名)
# 例如提取 Alpha, Beta, Gamma，可以命名为 ".../eeg_feature_bands/abg"
OUTPUT_ROOT = "/public/home/hugf2022/emotion/seediv/eeg_feature_bands/b"

SESSIONS = [1, 2, 3]

# === 核心配置：在这里定义你想要的频段 ===
# 可选值: 'delta', 'theta', 'alpha', 'beta', 'gamma'
# 例子1 (只要 Beta 和 Gamma): ['beta', 'gamma']
# 例子2 (只要 Alpha): ['alpha']
# 例子3 (要 Delta 和 Gamma): ['delta', 'gamma']
TARGET_BANDS = ['beta'] 

# ==========================================
# 2. Internal Logic (无需修改)
# ==========================================

# 频段名称到索引的映射
BAND_MAP = {
    'delta': 0,
    'theta': 1,
    'alpha': 2,
    'beta':  3,
    'gamma': 4
}

def get_band_indices(target_bands):
    """验证并获取目标频段的索引列表"""
    indices = []
    print(f"Target Bands: {target_bands}")
    try:
        for band in target_bands:
            indices.append(BAND_MAP[band.lower()])
    except KeyError as e:
        print(f"Error: Unknown band name {e}. Please use {list(BAND_MAP.keys())}")
        exit(1)
    
    # 打印映射结果供确认
    print(f"Mapped Indices: {indices}")
    return indices

def process_session(session_id, indices):
    input_dir = os.path.join(SOURCE_ROOT, str(session_id))
    output_dir = os.path.join(OUTPUT_ROOT, str(session_id))
    
    if not os.path.exists(input_dir):
        print(f"Skipping Session {session_id}: Input directory not found ({input_dir})")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"\nProcessing Session {session_id}...")
    print(f"  Source: {input_dir}")
    print(f"  Target: {output_dir}")
    
    files = [f for f in sorted(os.listdir(input_dir)) if f.endswith('.mat')]
    
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        save_path = os.path.join(output_dir, file_name)
        
        try:
            mat_data = scipy.io.loadmat(file_path)
        except Exception as e:
            print(f"    Error loading {file_name}: {e}")
            continue
            
        output_data = {}
        processed_count = 0
        
        # 遍历 .mat 文件中所有的 key
        for key in mat_data.keys():
            # 跳过系统 key
            if key.startswith('__'):
                continue
            
            data = mat_data[key]
            
            # 检查数据维度: (Channels, Time, 5) -> (62, T, 5)
            # 注意：这里的 5 是硬编码的，因为原始数据固定是5个频段
            if data.ndim == 3 and data.shape[2] == 5:
                # === 核心操作：根据索引列表切片 ===
                # 使用 numpy 的花式索引 (fancy indexing)
                new_data = data[:, :, indices]
                
                output_data[key] = new_data
                processed_count += 1
            else:
                # 如果有非特征矩阵的数据（如label），如果需要保留请取消注释
                # output_data[key] = data 
                pass

        if processed_count > 0:
            scipy.io.savemat(save_path, output_data)
            # 打印第一个数据的形状以供检查
            first_shape = list(output_data.values())[0].shape
            print(f"    Saved {file_name}: {processed_count} trials. New Shape: {first_shape}")
        else:
            print(f"    Warning: No valid feature data found in {file_name}")

def main():
    # 1. 获取目标索引
    target_indices = get_band_indices(TARGET_BANDS)
    
    # 2. 检查输出路径是否存在，防止误覆盖
    if os.path.exists(OUTPUT_ROOT) and os.listdir(OUTPUT_ROOT):
        print(f"\n[Warning] Output directory is not empty: {OUTPUT_ROOT}")
        print("Starting in 3 seconds... (Ctrl+C to cancel)")
        import time
        time.sleep(3)

    # 3. 开始处理
    for sess in SESSIONS:
        process_session(sess, target_indices)
        
    print("\nExtraction Complete!")
    print(f"Extracted bands: {TARGET_BANDS}")
    print(f"Check directory: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()