import torch
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.viz import plot_topomap
from mne_connectivity.viz import plot_connectivity_circle

# ==========================================
# 1. 配置电极名称 (保持之前修正后的顺序，包含 I1, I2)
# ==========================================
CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 
    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 
    'Oz', 'O2', 'FT9', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'FT10', 'TP9', 
    'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'TP10', 'PO5', 'PO3', 'PO4', 'PO6', 
    'P9', 'I1', 'I2', 'P10', 'F5', 'F1', 'F2', 'F6', 'P5', 'P1', 
    'P2', 'P6'
]

def create_info():
    """创建 MNE 的 Info 对象"""
    # 必须使用 'standard_1005' 才能支持 I1/I2 等扩展电极
    montage = mne.channels.make_standard_montage('standard_1005')
    
    info = mne.create_info(ch_names=CHANNEL_NAMES, sfreq=200, ch_types='eeg')
    
    try:
        info.set_montage(montage)
    except ValueError as e:
        print(f"Montage Warning: {e}")
        # 如果依然报错，可以选择忽略缺失的电极位置，虽然这会导致部分点画不出来
        # info.set_montage(montage, on_missing='ignore')
        
    return info

def plot_node_importance(node_imp, info, title="Node Importance"):
    """画脑地形图 (Topomap)"""
    fig, ax = plt.subplots(figsize=(5, 5))
    im, _ = plot_topomap(node_imp, info, axes=ax, show=False, 
                         cmap='Reds', names=CHANNEL_NAMES)
    plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, label='Importance')
    plt.title(title)
    return fig



def plot_edge_importance(edge_matrix, info, title="Connectivity", top_percent=5):
    """画大脑连接环形图 (Connectivity Circle)"""
    # 1. 阈值处理
    threshold = np.percentile(edge_matrix, 100 - top_percent)
    con = edge_matrix.copy()
    con[con < threshold] = 0 
    
    node_names = CHANNEL_NAMES
    
    # 2. 绘图
    # 修复点 1：删除了 text_color='black' 参数
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    plot_connectivity_circle(con, node_names, n_lines=None, 
                             node_angles=None, node_colors=None,
                             title=title, ax=ax, show=False, 
                             colormap='Blues', facecolor='white') 
    return fig

def main():
    # --------------------------------------------------
    # 1. 加载数据
    # --------------------------------------------------
    print("Loading results...")
    # 修复点 2：添加 weights_only=False 消除 Warning
    try:
        global_expl = torch.load("../results/dgcnn_global_explanation.pt", weights_only=False)
        static_adj = torch.load("../results/learned_static_adjacency.pt", weights_only=False)
    except FileNotFoundError:
        print("Error: 找不到结果文件，请检查路径 '../results/...' 是否正确")
        return

    # 转换为 numpy
    if isinstance(static_adj, torch.Tensor):
        static_adj = static_adj.numpy()
        
    info = create_info()
    
    # --------------------------------------------------
    # 2. 可视化：Learned Static Adjacency
    # --------------------------------------------------
    print("Plotting Learned Static Adjacency...")
    fig = plot_edge_importance(static_adj, info, title="DGCNN Learned Global Structure", top_percent=2)
    fig.savefig("../results/vis_static_adjacency.png", dpi=300)
    plt.show()

    # --------------------------------------------------
    # 3. 可视化：Group-level Saliency
    # --------------------------------------------------
    # 根据你的 SEED-IV 数据集，可能有 4 个类别 (0,1,2,3)
    emotions = {0: "Neutral", 1: "Sad", 2: "Fear", 3: "Happy"} 

    for class_idx, data in global_expl.items():
        emotion_name = emotions.get(class_idx, f"Class_{class_idx}")
        print(f"Plotting {emotion_name}...")
        
        node_imp = data["node_importance"]
        edge_imp = data["edge_importance"]
        
        # A. 画节点重要性
        fig_node = plot_node_importance(node_imp, info, title=f"{emotion_name} - Key Regions")
        fig_node.savefig(f"../results/vis_{emotion_name}_nodes.png", dpi=300)
        
        # B. 画边重要性
        fig_edge = plot_edge_importance(edge_imp, info, title=f"{emotion_name} - Key Connections", top_percent=1)
        fig_edge.savefig(f"../results/vis_{emotion_name}_edges.png", dpi=300)
        
        plt.show()

if __name__ == "__main__":
    main()