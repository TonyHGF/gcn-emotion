import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne
from mne.viz import plot_topomap
from mne_connectivity.viz import plot_connectivity_circle

# ==========================================
# 1. Configuration & Brain Region Definitions
# ==========================================

# Standard channel ordering for the SEED dataset (62 channels)
ORIG_CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F3', 'Fz', 'F4', 'F8', 
    'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 
    'CP1', 'CP2', 'CP6', 'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 
    'Oz', 'O2', 'FT9', 'FT7', 'FC3', 'FCz', 'FC4', 'FT8', 'FT10', 'TP9', 
    'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'TP10', 'PO5', 'PO3', 'PO4', 'PO6', 
    'P9', 'I1', 'I2', 'P10', 'F5', 'F1', 'F2', 'F6', 'P5', 'P1', 
    'P2', 'P6'
]

# Mapping brain regions to colors and naming conventions
# Format: (Region Label, RGB Color, Channel Prefix)
REGION_CONFIG = [
    ("Frontal",   (0.85, 0.3, 0.3),  ['F', 'AF']),       # Red
    ("Temporal",  (0.85, 0.85, 0.3), ['T', 'FT', 'TP']), # Yellow
    ("Central",   (0.3, 0.85, 0.3),  ['C', 'FC', 'CP']), # Green
    ("Parietal",  (0.3, 0.3, 0.85),  ['P', 'PO']),       # Blue
    ("Occipital", (0.6, 0.3, 0.6),   ['O', 'I', 'Oz'])   # Purple
]

def get_region_info(ch_name):
    """Assigns an electrode to a brain region and returns its label and color."""
    ch = ch_name.upper()
    for label, color, prefixes in REGION_CONFIG:
        for prefix in prefixes:
            if ch.startswith(prefix):
                return label, color
    return "Other", (0.7, 0.7, 0.7) # Default grey for undefined regions

def reorder_channels_by_region(adj_matrix, ch_names):
    """
    Reorders electrodes by anatomical region to ensure localized clustering 
    in the connectivity circle plot, preventing visual clutter.
    """
    # 1. Map regions to an integer order for sorting
    region_order_map = {cfg[0]: i for i, cfg in enumerate(REGION_CONFIG)}
    region_order_map["Other"] = 99
    
    sorted_indices = []
    sorted_names = []
    node_colors = []
    
    # Generate metadata for sorting
    temp_list = []
    for idx, name in enumerate(ch_names):
        region_label, color = get_region_info(name)
        order_key = region_order_map.get(region_label, 99)
        temp_list.append({
            'index': idx,
            'name': name,
            'color': color,
            'order': order_key,
            'label': region_label
        })
    
    # 2. Sort primary by region order, then secondary by channel name
    temp_list.sort(key=lambda x: (x['order'], x['name']))
    
    # 3. Extract sorted properties
    new_indices = [item['index'] for item in temp_list]
    sorted_names = [item['name'] for item in temp_list]
    node_colors = [item['color'] for item in temp_list]
    
    # 4. Permute the adjacency matrix (62x62) to match the new order
    new_adj = adj_matrix[new_indices, :][:, new_indices]
    
    return new_adj, sorted_names, node_colors

def create_info():
    """Initializes MNE Info object with standard 10-05 montage locations."""
    montage = mne.channels.make_standard_montage('standard_1005')
    info = mne.create_info(ch_names=ORIG_CHANNEL_NAMES, sfreq=200, ch_types='eeg')
    try:
        info.set_montage(montage)
    except ValueError:
        pass # Handle cases where some channel names don't match standard montage
    return info

# ==========================================
# 2. Visualization Functions
# ==========================================

def plot_node_importance_map(node_imp, info, title="Node Importance"):
    """Generates a topographic map showing the importance of each EEG channel."""
    fig, ax = plt.subplots(figsize=(5, 5))
    im, _ = plot_topomap(node_imp, info, axes=ax, show=False, 
                         cmap='Reds', names=ORIG_CHANNEL_NAMES)
    plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.6, label='Importance')
    plt.title(title)
    return fig

def plot_enhanced_circle(edge_matrix, info, title="Connectivity", top_percent=5):
    """
    Creates an enhanced connectivity circle plot with:
    1. Automated electrode clustering by brain region.
    2. Categorical color coding with a custom legend.
    """
    # --- Step A: Reorder data for anatomical clustering ---
    sorted_adj, sorted_names, sorted_colors = reorder_channels_by_region(edge_matrix, ORIG_CHANNEL_NAMES)
    
    # --- Step B: Thresholding (only show top X% strongest connections) ---
    threshold = np.percentile(sorted_adj, 100 - top_percent)
    con = sorted_adj.copy()
    con[con < threshold] = 0 
    
    # --- Step C: Plotting Connectivity Circle ---
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    plot_connectivity_circle(con, sorted_names, 
                             n_lines=None, 
                             node_angles=None, 
                             node_colors=sorted_colors, 
                             title=title, 
                             ax=ax, show=False, 
                             colormap='Blues', facecolor='white',
                             linewidth=1.0, 
                             fontsize_names=8, 
                             textcolor='black')
    
    # --- Step D: Add Region Legend ---
    legend_handles = []
    for label, color, _ in REGION_CONFIG:
        patch = mpatches.Patch(color=color, label=label)
        legend_handles.append(patch)
        
    # Use figure-level legend for precise placement outside the polar axes
    fig.legend(handles=legend_handles, loc='upper right', 
                bbox_to_anchor=(0.95, 0.95), title="Brain Regions", fontsize=10)
    
    return fig

# ==========================================
# 3. Execution Main Logic
# ==========================================
def main():
    print("Loading explanation results...")
    try:
        global_expl = torch.load("../results/dgcnn_global_explanation.pt", weights_only=False)
        static_adj = torch.load("../results/learned_static_adjacency.pt", weights_only=False)
    except FileNotFoundError:
        print("Error: Required .pt files not found in ../results/")
        return

    if isinstance(static_adj, torch.Tensor):
        static_adj = static_adj.numpy()
        
    info = create_info()
    
    # Task 1: Visualize the Learned Static Graph
    print("Plotting Static Circle...")
    fig = plot_enhanced_circle(static_adj, info, title="Static Connectivity Structure", top_percent=2)
    fig.savefig("../results/vis_static_circle_labeled.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Task 2: Visualize Emotion-Specific Explanation Graphs
    emotions = {0: "Neutral", 1: "Sad", 2: "Fear", 3: "Happy"} 
    for class_idx, data in global_expl.items():
        name = emotions.get(class_idx, f"Class_{class_idx}")
        print(f"Generating visualizations for: {name}...")
        
        # Topographic Node Importance Map
        fig_node = plot_node_importance_map(data["node_importance"], info, title=f"{name} - Regions")
        fig_node.savefig(f"../results/vis_{name}_nodes.png", dpi=300, bbox_inches='tight')
        
        # Clustered Edge Connectivity Circle
        fig_circle = plot_enhanced_circle(data["edge_importance"], info, title=f"{name} - Connections", top_percent=1)
        fig_circle.savefig(f"../results/vis_{name}_circle_labeled.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        plt.close('all')

if __name__ == "__main__":
    main()