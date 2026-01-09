# experiments/dgcnn_explain.py

import os
import torch
import logging
import matplotlib

# !!! 必须在导入 pyplot 之前设置后端，防止在服务器上报错
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from models import DGCNN
from explainer import DGCNNInterpreter
from data import SeedIVFeatureDataset

# 直接从你的 visualize_result.py 导入绘图函数，避免代码重复
from explainer.visualize_result import (
    create_info, 
    plot_node_importance_map, 
    plot_enhanced_circle
)

def run_dgcnn_explain(config: dict):
    logger = logging.getLogger("DGCNN-Explain")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --------------------------------------------------
    # 1. 准备数据与模型
    # --------------------------------------------------
    dataset = SeedIVFeatureDataset(
        root=config["data_root"],
        feature_key=config["feature_key"],
        sessions=config.get("test_sessions", [3]),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = DGCNN(
        num_electrodes=config["num_electrodes"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
    ).to(device)
    
    model.load_state_dict(torch.load(config["checkpoint_path"], map_location=device))
    
    # --------------------------------------------------
    # 2. 执行解释 (获取数值结果)
    # --------------------------------------------------
    explainer = DGCNNInterpreter(model, device=device)
    results = {}
    
    logger.info("[Explain] Calculating saliency maps...")
    for class_idx in range(config["num_classes"]):
        avg_node, avg_edge, sample_count = explainer.explain_group(loader, class_idx)
        if avg_node is not None:
            results[class_idx] = {
                "node_importance": avg_node,
                "edge_importance": avg_edge,
                "sample_count": sample_count,
            }
            logger.info(f" -> Class {class_idx}: {sample_count} samples explained.")

    # --------------------------------------------------
    # 3. 保存数值结果 (.pt)
    # --------------------------------------------------
    output_dir = config.get("output_dir", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(results, os.path.join(output_dir, "global_explanation.pt"))
    
    # 提取静态图结构 (如果有)
    global_adj = None
    if hasattr(model, "adjacency"):
        global_adj = (
            model.activation(model.adjacency + model.adjacency_bias)
            .detach().cpu().numpy()
        )
        torch.save(torch.tensor(global_adj), os.path.join(output_dir, "learned_static_adjacency.pt"))

    # --------------------------------------------------
    # 4. 调用 visualize_result 生成图片
    # --------------------------------------------------
    logger.info("[Explain] Generating visualizations...")
    
    try:
        # 获取 MNE Info 对象 (从你的代码里调用)
        info = create_info()
        emotions = {0: "Neutral", 1: "Sad", 2: "Fear", 3: "Happy"}

        # A. 画静态连接图
        if global_adj is not None:
            fig = plot_enhanced_circle(global_adj, info, title="Static Connectivity", top_percent=2)
            fig.savefig(os.path.join(output_dir, "vis_static_adj.png"), dpi=300, bbox_inches='tight')
            plt.close(fig) # 这一步很重要，释放内存

        # B. 画各类别的解释图
        for class_idx, data in results.items():
            name = emotions.get(class_idx, f"Class_{class_idx}")
            
            # 1. 节点重要性 (拓扑图)
            fig_node = plot_node_importance_map(data["node_importance"], info, title=f"{name} - Regions")
            fig_node.savefig(os.path.join(output_dir, f"vis_{name}_nodes.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_node)

            # 2. 边重要性 (圆环图)
            fig_circle = plot_enhanced_circle(data["edge_importance"], info, title=f"{name} - Connections", top_percent=1)
            fig_circle.savefig(os.path.join(output_dir, f"vis_{name}_circle.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_circle)
            
        logger.info(f"[Explain] All results and images saved to {output_dir}")

    except Exception as e:
        logger.error(f"[Explain] Visualization failed: {e}")
        logger.warning("Ensure 'mne' is installed and 'visualize_result.py' is in the explainer folder.")