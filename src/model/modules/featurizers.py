import torch
import torch.nn as nn
import torch.nn.functional as F

from src.SE3.se3_transformer.model import SE3Transformer
from src.SE3.se3_transformer.model.fiber import Fiber
from dgl.nn import EGATConv
from src.model.modules.layers import EGNNConv, NodeEGNNConv, AttentionEGNNConv


class Grid_SE3(nn.Module):
    """SE(3) equivariant GAT"""
    def __init__(self, num_layers_grid=2,
                 num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features=32,
                 l0_out_features=32,
                 l1_in_features=0,
                 l1_out_features=0,
                 num_edge_features=32, ntypes=15,
                 dropout_rate=0.1,
                 **kwargs):

        super().__init__()

        fiber_in = Fiber({0: l0_in_features}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features, 1: l1_in_features})

        self.se3 = SE3Transformer(
            num_layers   = num_layers_grid,
            num_heads    = n_heads,
            channels_div = 4,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: l0_out_features}), #1:l1_out_features}),
            fiber_edge=Fiber({0: num_edge_features}),
        )

        Cblock = []
        for i in range(ntypes):
            Cblock.append(nn.Linear(l0_out_features,1,bias=False))

        self.Cblock = nn.ModuleList(Cblock)
        self.dropoutlayer = nn.Dropout(dropout_rate)

    def forward(self, G, node_features, edge_features=None, drop_out=False):

        node_in, edge_in = {},{}
        for key in node_features:
            node_in[key] = node_features[key]
            if drop_out: node_in[key] = self.dropoutlayer(node_in[key])

        hs = self.se3(G, node_in, edge_features)

        hs0 = hs['0'].squeeze(2)
        if drop_out:
            hs0 = self.dropoutlayer(hs0)

        # hs0 as pre-FC embedding; N x num_channels
        cs = []
        for i,layer in enumerate(self.Cblock):
            c = layer(hs0)
            cs.append(c) # Nx2
        cs = torch.stack(cs,dim=0) #ntype x N x 2
        cs = cs.permute(1, 0, 2).squeeze(2)  # N x ntypes

        return hs0, cs


class Grid_EGNN(nn.Module):
    def __init__(self,
                 model='basic',
                 num_layers_grid=2,
                 l0_in_features=102,
                 l0_out_features=32,
                 num_channels=32,
                 num_edge_features=3,
                 ntypes=6,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__()
        
        self.num_layers = num_layers_grid
        self.update_coords = False  
        self.input_proj = nn.Sequential(
            nn.Linear(l0_in_features, num_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.egnn_layers = nn.ModuleList()
        for i in range(num_layers_grid):
            if i == 0:
                in_size = num_channels
            else:
                in_size = num_channels
                
            if model == 'egnn_coords': # update coordinates as well (as in gd4)
                self.update_coords = True
                layer = EGNNConv(
                    in_size=in_size,
                    hidden_size=num_channels,
                    out_size=num_channels,
                    dropout=dropout_rate,
                    edge_feat_size=num_edge_features
                )
            elif model == 'egnn_attention': # EGNN + attention
                layer = AttentionEGNNConv(
                    in_size=in_size,
                    hidden_size=num_channels,
                    out_size=num_channels,
                    dropout=dropout_rate,
                    edge_feat_size=num_edge_features
                )
            else:  # basic EGNN
                layer = NodeEGNNConv(
                    in_size=in_size,
                    hidden_size=num_channels,
                    out_size=num_channels,
                    dropout=dropout_rate,
                    edge_feat_size=num_edge_features
                )
            self.egnn_layers.append(layer)
        
        self.output_proj = nn.Sequential(
            nn.Linear(num_channels, l0_out_features),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Motif Label Classification = same as Grid_SE3 
        Cblock = []
        for i in range(ntypes):
            Cblock.append(nn.Linear(l0_out_features, 1, bias=False))
        self.Cblock = nn.ModuleList(Cblock)
        
        self.dropoutlayer = nn.Dropout(dropout_rate)

    def forward(self, G, node_features, edge_features=None, drop_out=False):
        """Forward pass - same as Grid_SE3
        
        Parameters
        ----------
        G : DGLGraph
            Grec
        node_features : dict
            Node features. keys: '0', 'x' 
        edge_features : dict, optional
            Edge features with key '0'
        drop_out : bool
            Whether to apply dropout
            
        Returns same as Grid_SE3
        -------
        hs0 : Updated node features of shape (N, l0_out_features)
        cs : Classification scores of shape (N, ntypes)
        """
        # Extract node features and coordinates
        h = node_features['0'].squeeze(2)  # [N, l0_in_features]
        coord = node_features['x'].squeeze()  # [N, 3] - remove all extra dims
        
        # ALSO DIFERENT FROM Grid_SE3!! Ensure coord is 2D: [N, 3]
        if coord.dim() > 2:
            coord = coord.squeeze(1)
        
        if drop_out:
            h = self.dropoutlayer(h)
        
        h = self.input_proj(h)
        
        edge_feat = None
        if edge_features is not None and '0' in edge_features:
            edge_feat = edge_features['0']
            # in msk1.py -> added dimension b/c Grid_SE3 uses 3D. squeeze b/c EGNN uses 2D
            if edge_feat.dim() > 2:
                edge_feat = edge_feat.squeeze(-1)  # rm SE(3) fiber dimension
        
        for i, layer in enumerate(self.egnn_layers):
            if drop_out:
                h = self.dropoutlayer(h)
                
            if self.update_coords:
                h, coord = layer(G, h, coord, edge_feat)
            else:
                h = layer(G, h, coord, edge_feat)
        
        hs0 = self.output_proj(h)
        
        if drop_out:
            hs0 = self.dropoutlayer(hs0)
        
        cs = []
        for i, layer in enumerate(self.Cblock):
            c = layer(hs0)
            cs.append(c)
        cs = torch.stack(cs, dim=0)
        cs = cs.permute(1, 0, 2).squeeze(2)  # N x ntypes
        
        return hs0, cs


class Ligand_SE3(nn.Module):
    """SE(3) equivariant GAT for ligands"""
    def __init__(self, num_layers=2,
                 num_channels=32,
                 num_degrees=3,
                 n_heads=4,
                 div=4,
                 l0_in_features=15,
                 l0_out_features=32,
                 l1_in_features=0,
                 l1_out_features=0,
                 num_edge_features=5, #(bondtype-1hot x4, d) -- ligand only
                 dropout_rate=0.1,
                 bias=True,
                 **kwargs):
        super().__init__()

        self.l1_in_features = l1_in_features
        self.dropoutlayer = nn.Dropout(p=dropout_rate)

        fiber_in = Fiber({0: l0_in_features}) if l1_in_features == 0 \
            else Fiber({0: l0_in_features, 1: l1_in_features})

        # processing ligands
        self.se3 = SE3Transformer(
            num_layers   = num_layers,
            num_heads    = n_heads,
            channels_div = div,
            fiber_in=fiber_in,
            fiber_hidden=Fiber({0: num_channels, 1:num_channels, 2:num_channels}),
            fiber_out=Fiber({0: l0_out_features}),
            fiber_edge=Fiber({0: num_edge_features}),
        )

    def forward(self, Glig, drop_out=True):
        node_features = {'0':Glig.ndata['attr'][:,:,None].float()}
        edge_features = {'0':Glig.edata['attr'][:,:,None].float()}

        if drop_out:
            node_features['0'] = self.dropoutlayer( node_features['0'] )

        hs = self.se3(Glig, node_features, edge_features)['0'] # M x d x 1

        return hs.squeeze(-1)

class Ligand_GAT(torch.nn.Module):
    def __init__(self,
                 l0_in_features,
                 num_edge_features, 
                 l0_out_features,
                 num_layers,
                 n_heads,
                 num_channels,
                 add_skip_connection=True,
                 bias=True,
                 dropout_rate=0.1,
                 **kwargs):

        super().__init__()

        # linear projection
        self.num_channels = num_channels
        self.n_heads = n_heads

        gat_layers = []
        #norm_layers = []
        for i in range(num_layers):
            '''layer = GATLayer(
                num_in_features=num_channels,
                num_out_features=num_channels,
                num_of_heads=n_heads,
                last_layer=(i==num_layers-1),
                add_skip_connection=add_skip_connection,
            )'''
            layer = EGATConv(in_node_feats=num_channels,
                             in_edge_feats=num_channels,
                             out_node_feats=num_channels,
                             out_edge_feats=num_channels,
                             num_heads=n_heads
            )
            #norm = nn.InstanceNorm1d(num_channels)
            gat_layers.append(layer)

        self.initial_linear = nn.Linear(l0_in_features, num_channels)
        self.initial_linear_edge = nn.Linear(num_edge_features, num_channels)

        self.dropout = nn.Dropout(p=dropout_rate)
        self.gat_net = nn.ModuleList( gat_layers )
        self.final_linear = nn.Linear(num_channels, l0_out_features)
        self.norm = nn.InstanceNorm1d(num_channels)

    def forward(self, Glig, drop_out=True):
        isolated = ((Glig.in_degrees()==0) & (Glig.out_degrees()==0)).nonzero().squeeze()
        Glig.remove_nodes(isolated)

        if drop_out:
            in_node_features = self.dropout(Glig.ndata['attr'])
            in_edge_features = self.dropout(Glig.edata['attr'])
        else:
            in_node_features = Glig.ndata['attr']
            in_edge_features = Glig.edata['attr']

        edge_index = Glig.edges()

        # project first
        emb0 = self.initial_linear( in_node_features )
        edge_emb0 = self.initial_linear_edge( in_edge_features )

        emb = emb0
        edge_emb = edge_emb0
        for i,layer in enumerate(self.gat_net):
            emb, edge_emb = layer( Glig, emb, edge_emb )
            emb = emb.mean(1)
            edge_emb = edge_emb.mean(1)

            emb = self.norm(emb)
            edge_emb = self.norm(edge_emb)

            emb = torch.nn.functional.elu(emb)
            edge_emb = torch.nn.functional.elu(edge_emb)

        out = self.final_linear( emb )
        return out
