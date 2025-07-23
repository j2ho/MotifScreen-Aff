# src/model/model.py

import sys
import torch
import torch.nn as nn

from src.model.modules.modules import StructModule, LigandModule, XformModule, DistanceModule 
from src.model.modules.classification import ClassModule
from src.model.modules.trigon import TrigonModule
from src.model.modules.featurizers import Grid_SE3, Ligand_SE3, Ligand_GAT 
from src.model.utils import to_dense_batch, make_batch_vec, get_pair_dis_one_hot, masked_softmax
    
class EndtoEndModel(nn.Module):
    """SE(3) equivariant GCN with attention"""
    def __init__(self, args):
        super().__init__()

        self.dropout_rate = args.dropout_rate

        self.lig_to_key_attn = args.params_TR.lig_to_key_attn
        self.shared_trigon = args.params_TR.shared_trigon
        
        m = args.params_TR.m            
        c = args.params_TR.c
        self.d = args.params_TR.c
        dropout_rate = args.params_TR.dropout_rate

        ## 1) Grid/Ligand featurizer
        self.GridFeaturizer = Grid_SE3( **args.params_grid.__dict__ )
            
        if args.params_ligand.model == 'se3':
            self.LigandFeaturizer = Ligand_SE3( **args.params_ligand.__dict__ )
        elif args.params_ligand.model == 'gat':
            self.LigandFeaturizer = Ligand_GAT( **args.params_ligand.__dict__ )
        else:
            sys.exit("unknown ligand_model: "+args.params_ligand.model)


        ## 2) Trigon-attn module
        self.trigon_lig = TrigonModule( args.params_TR.n_trigon_lig_layers,
                                        m, c, dropout_rate )
        

        ## 3) Heads
        self.class_module = ClassModule( m, c,
                                         args.params_Aff.classification_mode,
                                         args.params_ligand.n_lig_global_out)

        self.transform_distance = DistanceModule( c )
        self.struct_module = StructModule( c )
        self.extract_ligand_embedding = LigandModule( args.dropout_rate,
                                                      n_input=args.params_ligand.n_lig_global_in,
                                                      n_out=args.params_ligand.n_lig_global_out, )

        self.sig_Rl = torch.nn.Parameter(torch.tensor(10.0))

        Nlayers = args.params_TR.n_trigon_key_layers
        normalize = args.params_TR.normalize_Xform

        if self.shared_trigon:
            trigon_key_layer = TrigonModule(1, m, c, dropout_rate)
            self.trigon_key_layers = nn.ModuleList([ trigon_key_layer for _ in range(Nlayers) ])
            
        else:
            self.trigon_key_layers = nn.ModuleList([
                TrigonModule(1, m, c, dropout_rate) for _ in range(Nlayers)]) 
        self.XformKeys = nn.ModuleList([ XformModule( c, normalize=normalize ) for _ in range(Nlayers) ])
        self.XformGrids = nn.ModuleList([ XformModule( c, normalize=normalize ) for _ in range(Nlayers) ])

    def forward(self, Grec, Glig, keyidx, grididx, u=None,
                gradient_checkpoint=True, drop_out=False):

        # 1) first process Grec to get h_rec -- "motif"-embedding
        node_features = {'0':Grec.ndata['attr'][:,:,None].float(), 'x': Grec.ndata['x'].float() }
        edge_features = {'0':Grec.edata['attr'][:,:,None].float()}

        h_rec, cs = self.GridFeaturizer(Grec, node_features, edge_features, drop_out)
        # print(f"DEBUG: GridFeaturizer output - h_rec shape: {h_rec.shape if h_rec is not None else 'None'}, cs shape: {cs.shape if cs is not None else 'None'}")

        gridmap = torch.eye(h_rec.shape[0]).to(Grec.device)[grididx]

        h_grid = torch.matmul(gridmap, h_rec) # grid part
        # print(f"DEBUG: h_grid shape: {h_grid.shape if h_grid is not None else 'None'}")
        
        # 1-1) trim to grid part of Grec
        Ggrid = Grec.subgraph( grididx )
        NullArgs = (None, None, None, None, None)

        if (Ggrid.batch_num_nodes()==0).any():
            return NullArgs

        Ykey_s, z_norm, aff = None, None, None
        if Glig is None: # if no ligand info provided
            return NullArgs

        # 2) ligand embedding
        try:
            h_lig = self.LigandFeaturizer(Glig, drop_out=drop_out)
            # print(f"DEBUG: LigandFeaturizer output - h_lig shape: {h_lig.shape if h_lig is not None else 'None'}")
        except Exception as e:
            # print(f"DEBUG: LigandFeaturizer failed with error: {e}")
            return NullArgs

        # global embedding if needed
        h_lig_global = self.extract_ligand_embedding( Glig.gdata.to(Glig.device),
                                                      dropout=drop_out ) # gdata isn't std attr
        # print(f"DEBUG: extract_ligand_embedding output - h_lig_global shape: {h_lig_global.shape if h_lig_global is not None else 'None'}")

        # 3) Prep Trigon attention
        # 3-1) Grid part
        batchvec_grid = make_batch_vec(Ggrid.batch_num_nodes()).to(Ggrid.device)
        gridxyz = Ggrid.ndata['x'].squeeze().float()
        grid_x_batched, grid_mask = to_dense_batch(gridxyz, batchvec_grid)
        D_grid = get_pair_dis_one_hot(grid_x_batched, bin_size=0.25, bin_min=-0.1, bin_max=15.75, num_classes=self.d).float()
        h_grid_batched, _ = to_dense_batch(h_grid, batchvec_grid)

        # 3-2) ligand part
        batchvec_lig = make_batch_vec(Glig.batch_num_nodes()).to(Grec.device)
        ligxyz = Glig.ndata['x'].squeeze().float()
        lig_x_batched, lig_mask = to_dense_batch(ligxyz, batchvec_lig)
        D_lig  = get_pair_dis_one_hot(lig_x_batched, bin_size=0.25, bin_min=-0.1, bin_max=15.75, num_classes=self.d).float()
        h_lig_batched, _  = to_dense_batch(h_lig, batchvec_lig)
        # vars up to here

        # 3-3) trigon1 "pre-keying"
        z_mask = torch.einsum('bn,bm->bnm', grid_mask, lig_mask )

        z = self.trigon_lig( h_grid_batched, h_lig_batched, z_mask,
                             D_grid, D_lig,
                             drop_out=drop_out )
        # print(f"DEBUG: trigon_lig output - z shape: {z.shape if z is not None else 'None'}")

        # Ligand-> Key mapper (trim down ligand -> key)
        Kmax = max([idx.shape[0] for idx in keyidx])
        Nmax = max([idx.shape[1] for idx in keyidx])
        key_idx = torch.zeros((Glig.batch_num_nodes().shape[0],Kmax,Nmax)).to(Grec.device) #b x K x N
        lig_to_key_mask = torch.zeros_like( key_idx ) # B x K x N
        for i,idx in enumerate(keyidx):
            key_idx[i,:idx.shape[0],:idx.shape[1]] = idx
            lig_to_key_mask[i,:idx.shape[0],:idx.shape[1]] = 1.0

        # trim down to key after trigon
        # key_idx: B x K x M
        key_x_batched = torch.einsum('bik,bji->bjk', lig_x_batched, key_idx)
        
        h_key_batched = torch.einsum('bkj,bjd->bkd',key_idx,h_lig_batched)

        if self.lig_to_key_attn:
            # key: h_key, query: h_lig
            A = torch.einsum('bkd,bjd->bkj', h_key_batched, h_lig_batched)
            A = masked_softmax( A, lig_to_key_mask, dim=2 ) # softmax over N
            
            h_key_batched = h_key_batched + torch.einsum('bkj,bjd->bkd', A, h_lig_batched )
            h_key_batched = nn.functional.layer_norm(h_key_batched, h_key_batched.shape)
        
        D_key1  = get_pair_dis_one_hot(key_x_batched, bin_size=0.25, bin_min=-0.1, bin_max=15.75, num_classes=self.d).float()
        
        # vars up to here
        z = torch.einsum( 'bkj,bijd->bikd', key_idx, z)

        # 3-4) key-position-aware attn
        # shared params; sort of "recycle"
        key_mask = torch.einsum('bkj,bj->bk', key_idx, lig_mask.float()).bool()
        z_mask = torch.einsum('bn,bm->bnm', grid_mask, key_mask )

        #TODO
        h_grid_batched = h_grid_batched.repeat(h_key_batched.shape[0],1,1)
        for trigon,xformK,xformG in zip(self.trigon_key_layers,self.XformKeys,self.XformGrids):
            
            # move from below to here so that h shares embedding w/ structure...
            # would it make difference?
            # update key/grid features using learned attention
            D_key = self.transform_distance( h_key_batched )
            
            h_key_batched  = xformK( h_key_batched, h_grid_batched, z, z_mask, dim=2 ) # key/query/attn
            h_grid_batched = xformG( h_grid_batched, h_key_batched, z, z_mask, dim=1 ) # key/query/attn
            # update z
            z = trigon(h_grid_batched, h_key_batched, z_mask,
                       D_grid, D_key,
                       drop_out=drop_out )
            # print(f"DEBUG: trigon layer output - z shape: {z.shape if z is not None else 'None'}")

            #z_mask = torch.einsum('bn,bm->bnm', grid_mask, key_mask )
            # Ykey_s: B x K x 3; z_norm: B x N x K x d
            # z: B x N x K x d; z_maks: B x N x K
            Ykey_s, z_norm = self.struct_module( z, z_mask, Ggrid, key_mask )
            # print(f"DEBUG: struct_module output - Ykey_s shape: {Ykey_s.shape if Ykey_s is not None else 'None'}, z_norm shape: {z_norm.shape if z_norm is not None else 'None'}")

            #D_key = update_D_from_Ypred( Ykey_s, num_classes=self.d )

        # 2-2) screening module
        aff = self.class_module( z, h_grid_batched, h_key_batched,
                                 lig_rep=h_lig_global, w_mask=key_mask )
        
        return Ykey_s, D_key, z_norm, cs, aff