import torch
import torch.nn as nn
import numpy as np

from .layer_pgcn import LocalLayer, MesoLayer, GlobalLayer
from .node_location import get_ini_dis_m, convert_dis_m, return_coordinates

def normalize_adj(adj):
    """
    Implements the normalize_adj function from your utils.py
    Logic: D^-0.5 * A * D^-0.5
    """
    degree = torch.sum(adj, dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.matmul(torch.matmul(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)

class PGCN(nn.Module):
    """
    PGCN aligned with DGCNN input/output.
    Original structure preserved from model_PGCN.py
    """
    def __init__(
        self, 
        num_electrodes: int = 62, 
        in_channels: int = 5, 
        num_classes: int = 3, 
        dropout_rate: float = 0.5,
        lr: float = 0.1 
    ):
        super(PGCN, self).__init__()
        
        self.dropout = dropout_rate
        self.l_relu = lr
        
        # ---------------------------------------------------------------
        # 1. Graph Initialization (Matches __init__ setup)
        # ---------------------------------------------------------------
        # Instead of passing 'local_adj' and 'coor' arguments, we build them here
        # so the model self-contained like your DGCNN.
        dis_m = get_ini_dis_m()       
        adj_np = convert_dis_m(dis_m) 
        coor_np = return_coordinates() 
        
        # Register as buffers (moves to GPU automatically)
        self.register_buffer('adj', torch.from_numpy(adj_np).float())
        self.register_buffer('coordinate', torch.from_numpy(coor_np).float())

        # ---------------------------------------------------------------
        # 2. Local GCN (Matches lines 29-30 in model_PGCN.py)
        # ---------------------------------------------------------------
        # Original: self.local_gcn_1 = LocalLayer(args.in_feature, 10, True)
        self.local_gcn_1 = LocalLayer(in_channels, 10, True)
        # Original: self.local_gcn_2 = LocalLayer(10, 15, True)
        self.local_gcn_2 = LocalLayer(10, 15, True)

        # ---------------------------------------------------------------
        # 3. Meso Scale (Matches lines 38-42 in model_PGCN.py)
        # ---------------------------------------------------------------
        # Original: self.meso_embed = nn.Linear(5, 30)
        self.meso_embed = nn.Linear(in_channels, 30) 
        
        # Original: MesoLayer(subgraph_num=7, num_heads=6, ..., trainable_vector=78)
        self.meso_layer_1 = MesoLayer(
            subgraph_num=7, 
            num_heads=6, 
            coordinate=self.coordinate, 
            trainable_vector=78
        )
        # Original: MesoLayer(subgraph_num=2, num_heads=6, ..., trainable_vector=54)
        self.meso_layer_2 = MesoLayer(
            subgraph_num=2, 
            num_heads=6, 
            coordinate=self.coordinate, 
            trainable_vector=54
        )
        self.meso_dropout = nn.Dropout(0.2) # Kept from original line 42

        # ---------------------------------------------------------------
        # 4. Global Scale (Matches lines 47 in model_PGCN.py)
        # ---------------------------------------------------------------
        # Original: self.global_layer_1 = GlobalLayer(30, 40)
        self.global_layer_1 = GlobalLayer(30, 40)

        # ---------------------------------------------------------------
        # 5. MLP / Classifier (Matches lines 65-67 in model_PGCN.py)
        # ---------------------------------------------------------------
        # Calculation validation:
        # Local output dim = 5 (input) + 10 (gcn1) + 15 (gcn2) = 30 features
        # Meso output nodes = 62 (local) + 7 (meso1) + 2 (meso2) = 71 nodes
        # Global output dim = 30 (meso_in) + 40 (global_out) = 70 features
        # Final flatten = 71 * 70
        
        self.mlp0 = nn.Linear(71*70, 2048)
        self.mlp1 = nn.Linear(2048, 1024)
        self.mlp2 = nn.Linear(1024, num_classes)

        # Common layers (Matches lines 71-73)
        self.bn = nn.BatchNorm1d(1024)
        self.lrelu = nn.LeakyReLU(self.l_relu)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        Input x: (B, 62, 5)
        """
        # Ensure submodule coordinates match model device
        self.meso_layer_1.coordinate = self.coordinate
        self.meso_layer_2.coordinate = self.coordinate

        # ---------------------------------------------------------------
        # Step 1: Local GCN (Matches lines 82-87)
        # ---------------------------------------------------------------
        laplacian = normalize_adj(self.adj)

        local_x1 = self.lrelu(self.local_gcn_1(x, laplacian, True))
        local_x2 = self.lrelu(self.local_gcn_2(local_x1, laplacian, True))
        
        # Concatenate on dim 2 (features): 5 + 10 + 15 = 30
        res_local = torch.cat((x, local_x1, local_x2), 2) 

        # ---------------------------------------------------------------
        # Step 2: Meso Scale (Matches lines 94-105)
        # ---------------------------------------------------------------
        meso_input = self.meso_embed(x) 
        
        coarsen_x1, coarsen_coor1 = self.meso_layer_1(meso_input)
        coarsen_x1 = self.lrelu(coarsen_x1)

        coarsen_x2, coarsen_coor2 = self.meso_layer_2(meso_input)
        coarsen_x2 = self.lrelu(coarsen_x2)

        # Concatenate on dim 1 (nodes): 62 + 7 + 2 = 71 nodes
        res_meso = torch.cat((res_local, coarsen_x1, coarsen_x2), 1)
        res_coor = torch.cat((self.coordinate, coarsen_coor1, coarsen_coor2), 0)

        # ---------------------------------------------------------------
        # Step 3: Global Scale (Matches lines 117-121)
        # ---------------------------------------------------------------
        global_x1 = self.lrelu(self.global_layer_1(res_meso, res_coor))
        
        # Concatenate on dim 2 (features): 30 (res_meso) + 40 (global) = 70 features
        res_global = torch.cat((res_meso, global_x1), 2)

        # ---------------------------------------------------------------
        # Step 4: Emotion Recognition (Matches lines 130-138)
        # ---------------------------------------------------------------
        x_flat = res_global.view(res_global.size(0), -1)

        x_out = self.lrelu(self.mlp0(x_flat))
        x_out = self.dropout_layer(x_out)
        
        # Note: Original code had bn commented out here
        
        x_out = self.lrelu(self.mlp1(x_out))
        x_out = self.bn(x_out)
        
        # Note: Original code had dropout commented out here
        
        logits = self.mlp2(x_out)

        return logits