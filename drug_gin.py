"""
PathDSP-X Drug Graph Isomorphism Network (DrugGIN)
==================================================
This module implements a 4-layer Graph Isomorphism Network with edge features
(using PyTorch Geometric's GINEConv) and attention-based global pooling to 
produce a 128-dimensional molecular embedding from molecular graphs.

Mathematical Formulation:
-------------------------
For each GINE layer k:
    h_v^(k) = MLP^(k) ( (1 + eps^(k)) * h_v^(k-1) + sum_{u in N(v)} ReLU(h_u^(k-1) + e_{vu}^(k-1)) )

- eps^(k) is a learnable parameter per layer (train_eps=True).
- Batch normalization and ReLU activations are applied between layers.
- Atom representations are aggregated using Attention-Based Global Pooling:
    s_v = softmax_v ( gate_nn(h_v) )
    h_graph = sum_v ( s_v * h_v )
- Output dimension: 128-dim graph embedding vector.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv

# Import attention pooling with backward/forward compatibility
try:
    from torch_geometric.nn.aggr import AttentionalAggregation
    HAS_ATTENTIONAL_AGGR = True
except ImportError:
    HAS_ATTENTIONAL_AGGR = False

from torch_geometric.nn import GlobalAttention


class DrugGIN(nn.Module):
    """
    4-layer Graph Isomorphism Network with Edge Features (GINEConv)
    and Attention-based Global Pooling for molecular drug embeddings.

    Args:
        node_in_dim: Number of input node (atom) features (default: 10).
        edge_in_dim: Number of input edge (bond) features (default: 6).
        hidden_dim: Hidden dimension across GIN layers (default: 128).
        out_dim: Final drug embedding dimension (default: 128).
        num_layers: Number of GINEConv message passing layers (default: 4).
        dropout: Dropout rate applied after each layer (default: 0.1).
        train_eps: Whether epsilon is learnable in GINEConv (default: True).
    """
    def __init__(
        self,
        node_in_dim: int = 10,
        edge_in_dim: int = 6,
        hidden_dim: int = 128,
        out_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        train_eps: bool = True
    ):
        super().__init__()
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Initial linear encoders to project atom and bond features to hidden_dim
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        # 4-layer GINEConv stack
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            # 2-layer MLP for GINEConv: (hidden_dim -> hidden_dim -> hidden_dim)
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # GINEConv with learnable epsilon and edge_dim matching hidden_dim
            conv = GINEConv(mlp, eps=0.0, train_eps=train_eps, edge_dim=hidden_dim)
            self.convs.append(conv)
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Attention-based Global Pooling
        # Soft attention weights gate_nn(h_v) compute importance of each atom in the molecule
        gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        if HAS_ATTENTIONAL_AGGR:
            self.pool = AttentionalAggregation(gate_nn=gate_nn)
            self._use_aggr = True
        else:
            self.pool = GlobalAttention(gate_nn=gate_nn)
            self._use_aggr = False

        # Output projection head to guarantee out_dim (128)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim) if out_dim == hidden_dim else nn.Identity()
        )

    def forward(
        self,
        x: Union[torch.Tensor, Data, Batch],
        edge_index: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Can be invoked with either:
          1. A PyG Data or Batch object as the first parameter:
             `out = model(batch_data)`
          2. Explicit tensors:
             `out = model(x, edge_index, edge_attr, batch)`

        Returns:
            torch.Tensor: Molecular embeddings of shape [batch_size, out_dim].
        """
        if hasattr(x, 'edge_index'):
            # PyG Data or Batch object
            data = x
            x = data.x
            edge_index = data.edge_index
            edge_attr = getattr(data, 'edge_attr', None)
            batch = getattr(data, 'batch', None)

        if batch is None:
            # Single graph: batch indices are all zeros
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # 1. Project input node features
        h = self.node_encoder(x)

        # 2. Project edge features
        if edge_attr is not None and edge_attr.numel() > 0:
            e = self.edge_encoder(edge_attr)
        else:
            # Zero-edges fallback (e.g. isolated atom / single ion molecule)
            e = torch.zeros((edge_index.size(1), self.hidden_dim), dtype=torch.float, device=x.device)

        # 3. Message passing through GINEConv layers with BatchNorm and ReLU
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index, e)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # 4. Attention-based global pooling
        if self._use_aggr:
            # AttentionalAggregation accepts index=batch
            graph_emb = self.pool(h, index=batch)
        else:
            # GlobalAttention accepts batch
            graph_emb = self.pool(h, batch)

        # 5. Output projection to final embedding dimension
        out = self.out_proj(graph_emb)
        return out


if __name__ == '__main__':
    # Quick test instantiation and forward pass
    print("Testing DrugGIN module...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DrugGIN(node_in_dim=10, edge_in_dim=6, hidden_dim=128, out_dim=128, num_layers=4).to(device)
    model.eval()

    # Synthetic sample graph
    x_sample = torch.randn(14, 10, device=device)
    edge_idx_sample = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long, device=device)
    edge_attr_sample = torch.randn(6, 6, device=device)
    batch_idx_sample = torch.zeros(14, dtype=torch.long, device=device)

    with torch.no_grad():
        out = model(x_sample, edge_idx_sample, edge_attr_sample, batch_idx_sample)

    print(f"DrugGIN instantiated successfully!")
    print(f"Output embedding shape: {out.shape} (Expected: [1, 128])")
