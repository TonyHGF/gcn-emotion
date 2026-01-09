import os
import numpy as np
import scipy.io

# ==========================================
# 1. Configuration (请修改路径)
# ==========================================

# ！！！这里填你【已经跑出来的、包含5个频段】的数据文件夹路径！！！
SOURCE_ROOT = "/public/home/hugf2022/emotion/seediv/m_eeg_feature_smooth" 

# 这里填你想要保存【只有 Beta/Gamma】的新路径
OUTPUT_ROOT = "/public/home/hugf2022/emotion/seediv/eeg_feature_bands/bg"

SESSIONS = [1, 2, 3]

# 原始数据的频段顺序 (仅作参考，用于确定索引)
# 0: delta, 1: theta, 2: alpha, 3: beta, 4: gamma
# 我们需要提取的是 index 3 和 4

def process_session(session_id):
    input_dir = os.path.join(SOURCE_ROOT, str(session_id))
    output_dir = os.path.join(OUTPUT_ROOT, str(session_id))
    
    if not os.path.exists(input_dir):
        print(f"Skipping Session {session_id}: Input directory not found ({input_dir})")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Processing Session {session_id}...")
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
            # 跳过系统生成的 key (__header__, __version__, __globals__)
            if key.startswith('__'):
                continue
            
            # 假设特征数据的 key 格式是 "de_LDS1", "de_LDS2" 等
            # 或者你之前的代码可能存的是其他名字，这里统一处理所有非系统 key
            data = mat_data[key]
            
            # 检查数据维度
            # 预期输入形状: (Channels, Time, 5) -> (62, T, 5)
            if data.ndim == 3 and data.shape[2] == 5:
                # === 核心操作：切片取最后两个频段 (Beta, Gamma) ===
                # index 3 是 Beta, index 4 是 Gamma
                new_data = data[:, :, 3:] 
                
                # 存入字典
                output_data[key] = new_data
                processed_count += 1
            else:
                # 如果遇到形状不对的数据（比如 label 或者其他信息），如果需要保留，可以取消下面注释
                # output_data[key] = data 
                pass

        if processed_count > 0:
            scipy.io.savemat(save_path, output_data)
            print(f"    Saved {file_name}: {processed_count} trials processed. Shape -> {list(output_data.values())[0].shape}")
        else:
            print(f"    Warning: No valid feature data found in {file_name}")

def main():
    for sess in SESSIONS:
        process_session(sess)
    print("\nExtraction Complete! Check directory:", OUTPUT_ROOT)

if __name__ == "__main__":
    main()