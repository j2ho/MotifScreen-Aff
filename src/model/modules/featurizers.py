import torch
import torch.nn as nn

from src.SE3.se3_transformer.model import SE3Transformer
from src.SE3.se3_transformer.model.fiber import Fiber
from dgl.nn import EGATConv


class Grid_SE3(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, num_layers_grid=2,
                 num_channels=32, num_degrees=3, n_heads=4, div=4,
                 l0_in_features=32,
                 l0_out_features=32,
                 l1_in_features=0,
                 l1_out_features=0,
                 num_edge_features=32, ntypes=15,
                 dropout_rate=0.1 ):

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
        cs = cs.T.squeeze(0)

        return hs0, cs

class Ligand_SE3(nn.Module):
    """SE(3) equivariant GCN with attention"""
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
                 bias=True):
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
                 num_edge_features, ## unused... can we use?
                 l0_out_features,
                 num_layers,
                 n_heads,
                 num_channels,
                 add_skip_connection=True,
                 bias=True,
                 dropout_rate=0.1):

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

        #emb0 = emb0.view(-1, self.n_heads, self.num_channels)
        #edge_emb = edge_emb.view(-1, self.n_heads, self.num_channels)

        emb = emb0
        edge_emb = edge_emb0
        for i,layer in enumerate(self.gat_net):
            emb, edge_emb = layer( Glig, emb, edge_emb )
            # should aggregate head dim
            # mean pooling
            emb = emb.mean(1)
            edge_emb = edge_emb.mean(1)

            emb = self.norm(emb)
            edge_emb = self.norm(edge_emb)

            emb = torch.nn.functional.elu(emb)
            edge_emb = torch.nn.functional.elu(edge_emb)

        # off if using dgl.nn
        out = self.final_linear( emb )
        return out
