"""EGNN and other equivariant layers for MotifScreen-Aff"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax


class EGNNConv(nn.Module):
    """E(n) Equivariant Graph Convolutional Layer
    
    From "E(n) Equivariant Graph Neural Networks" https://arxiv.org/abs/2102.09844
    
    Updates both node features and coordinates:
    - h_i^{l+1} = phi_h(h_i^l, sum_j m_ij)  
    - x_i^{l+1} = x_i^l + sum_j (x_i^l - x_j^l) * phi_x(m_ij)
    - m_ij = phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_ij)
    """
    
    def __init__(self, in_size, hidden_size, out_size, dropout=0.0, edge_feat_size=0):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # Edge MLP: phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the distance feature: ||x_i - x_j||
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # Node MLP: phi_h  
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, out_size),
        )

        # Coordinate MLP: phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def message(self, edges):
        """Message function for EGNN"""
        # Concatenate features for edge MLP
        if self.edge_feat_size > 0:
            f = torch.cat([
                edges.src["h"],
                edges.dst["h"], 
                edges.data["dist"],
                edges.data["a"],
            ], dim=-1)
        else:
            f = torch.cat([
                edges.src["h"], 
                edges.dst["h"], 
                edges.data["dist"]
            ], dim=-1)

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        """
        Parameters
        ----------
        graph : DGLGraph
            The graph
        node_feat : torch.Tensor
            Node features of shape (N, in_size)
        coord_feat : torch.Tensor  
            Coordinate features of shape (N, 3)
        edge_feat : torch.Tensor, optional
            Edge features of shape (M, edge_feat_size)
            
        Returns
        -------
        node_feat_out : torch.Tensor
            Updated node features of shape (N, out_size)
        coord_feat_out : torch.Tensor
            Updated coordinates of shape (N, 3)
        """
        with graph.local_scope():
            # Store features in graph
            graph.ndata["h"] = node_feat
            graph.ndata["x"] = coord_feat
            
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
                
            # Compute coordinate differences and distances
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["dist"] = (
                graph.edata["x_diff"].square().sum(dim=1).sqrt().unsqueeze(-1)
            )
            
            # Normalize coordinate differences
            graph.edata["x_diff"] = graph.edata["x_diff"] / (
                graph.edata["dist"] + 1e-7
            )
            
            # Apply message function
            graph.apply_edges(self.message)
            
            # Aggregate messages
            graph.update_all(fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh"))
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))

            h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

            # Update node features
            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))

            return h, coord_feat + x_neigh


class NodeEGNNConv(nn.Module):
    """EGNN layer that only updates node features (coordinates unchanged)"""
    
    def __init__(self, in_size, hidden_size, out_size, dropout=0.0, edge_feat_size=0):
        super(NodeEGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # Edge MLP: phi_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # Node MLP: phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, out_size),
        )

    def message(self, edges):
        """Message function - only for node updates"""
        if self.edge_feat_size > 0:
            # Debug: check shapes
            edge_attr = edges.data["a"]
            if edge_attr.dim() > 2:
                edge_attr = edge_attr.squeeze(-1)
                
            f = torch.cat([
                edges.src["h"],
                edges.dst["h"],
                edges.data["dist"],
                edge_attr,
            ], dim=-1)
        else:
            f = torch.cat([
                edges.src["h"], 
                edges.dst["h"], 
                edges.data["dist"]
            ], dim=-1)

        msg_h = self.edge_mlp(f)
        return {"msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        """Forward pass - coordinates unchanged"""
        with graph.local_scope():
            graph.ndata["h"] = node_feat
            graph.ndata["x"] = coord_feat
            
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
                
            # Compute distances  
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["dist"] = (
                graph.edata["x_diff"].square().sum(dim=1).sqrt().unsqueeze(-1)
            )

            graph.apply_edges(self.message)
            graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))

            h_neigh = graph.ndata["h_neigh"]
            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))

            return h


class AttentionEGNNConv(nn.Module):
    """EGNN with attention mechanism for node feature updates"""
    
    def __init__(self, in_size, hidden_size, out_size, dropout=0.0, edge_feat_size=0):
        super(AttentionEGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # Edge MLP: phi_e
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # Node MLP: phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, out_size),
        )
        
        # Attention components
        self.linear_q = nn.Linear(in_size, hidden_size, bias=False)
        self.linear_k = nn.Linear(in_size + edge_feat_size + 1, hidden_size, bias=False)

    def message(self, edges):
        """Message function with attention"""
        if self.edge_feat_size > 0:
            # Handle dimension mismatch from SE(3) input format
            edge_attr = edges.data["a"]
            if edge_attr.dim() > 2:
                edge_attr = edge_attr.squeeze(-1)
                
            f = torch.cat([
                edges.src["h"],
                edges.dst["h"],
                edges.data["dist"],
                edge_attr,
            ], dim=-1)
            
            k_raw = torch.cat([
                edges.src["h"],
                edges.data["dist"],
                edge_attr,
            ], dim=-1)
        else:
            f = torch.cat([
                edges.src["h"], 
                edges.dst["h"], 
                edges.data["dist"]
            ], dim=-1)
            
            k_raw = torch.cat([
                edges.src["h"],
                edges.data["dist"],
            ], dim=-1)

        msg_h = self.edge_mlp(f)
        
        # Attention computation
        msg_k = self.linear_k(k_raw)
        msg_q = self.linear_q(edges.dst["h"])
        msg_e = (msg_k * msg_q).sum(dim=-1) / math.sqrt(self.hidden_size)

        return {"msg_h": msg_h, 'e': msg_e.unsqueeze(-1)}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):
        """Forward pass with attention-weighted node feature updates"""
        with graph.local_scope():
            graph.ndata["h"] = node_feat
            graph.ndata["x"] = coord_feat
            
            if self.edge_feat_size > 0:
                assert edge_feat is not None, "Edge features must be provided."
                graph.edata["a"] = edge_feat
                
            # Compute distances for edge features
            graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
            graph.edata["dist"] = (
                graph.edata["x_diff"].square().sum(dim=1).sqrt().unsqueeze(-1)
            )
            
            graph.apply_edges(self.message)
            
            # Apply attention weights to node messages
            graph.edata['att'] = edge_softmax(graph, graph.edata['e']).unsqueeze(dim=-1)
            graph.edata['msg_f'] = graph.edata.pop('msg_h') * graph.edata.pop('att')
            graph.update_all(fn.copy_e("msg_f", "m"), fn.sum("m", "h_neigh"))

            h_neigh = graph.ndata["h_neigh"]
            h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
            
            return h