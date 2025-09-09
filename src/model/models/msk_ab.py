# src/model/models/msk_ab.py

import sys
import torch
import torch.nn as nn

from src.model.modules.modules import StructModule, LigandModule, XformModule, DistanceModule
from src.model.modules.classification import ClassModule
from src.model.modules.trigon import TrigonModule
from src.model.modules.featurizers import Grid_SE3, Grid_EGNN, Ligand_SE3, Ligand_GAT
from src.model.utils import to_dense_batch, make_batch_vec, get_pair_dis_one_hot, masked_softmax


class EndtoEndModel(nn.Module):
    """SE(3) equivariant GNNs with attention"""
    def __init__(self, args):
        super().__init__()

        self.dropout_rate = args.dropout_rate

        self.c = args.model_params_TR.c # embedding channels
        self.d = args.model_params_TR.d # distance channels(bins)
        self.dropout_rate = args.model_params_TR.dropout_rate
        
        self.ablation = args.ablation

        ## Grid/Ligand featurizer
        if args.model_params_grid.model == 'se3':
            self.GridFeaturizer = Grid_SE3( **args.model_params_grid.__dict__ )
        elif args.model_params_grid.model.startswith('egnn'):
            self.GridFeaturizer = Grid_EGNN( **args.model_params_grid.__dict__ )
        else:
            sys.exit("unknown grid_model: "+args.model_params_grid.model)

        if args.model_params_ligand.model == 'se3':
            self.LigandFeaturizer = Ligand_SE3( **args.model_params_ligand.__dict__ )
        elif args.model_params_ligand.model == 'gat':
            self.LigandFeaturizer = Ligand_GAT( **args.model_params_ligand.__dict__ )
        else:
            sys.exit("unknown ligand_model: "+args.model_params_ligand.model)


        self.extract_ligand_embedding = LigandModule( args.dropout_rate,
                                                      n_input=args.model_params_ligand.n_lig_global_in,
                                                      n_out=args.model_params_ligand.n_lig_global_out)


        ## Trigon-attn module
        # trigon module before masking only to key atoms
        if self.ablation.enable_trigon_lig:
            n_trigon_lig_layers = (
                self.ablation.num_trigon_lig_layers_override 
                if self.ablation.num_trigon_lig_layers_override is not None 
                else args.model_params_TR.n_trigon_lig_layers
            )
            self.trigon_lig = TrigonModule( n_trigonometry_module_stack=n_trigon_lig_layers,
                                            grid_m=args.model_params_grid.l0_out_features,
                                            ligand_m=args.model_params_ligand.l0_out_features,
                                            c=self.c, dropout_rate=self.dropout_rate )
        else:
            self.trigon_lig = None

        if self.ablation.enable_distance_transform:
            self.transform_distance = DistanceModule(self.d,self.c)
        else:
            self.transform_distance = None
            
        if self.ablation.enable_structure_module:
            self.struct_module = StructModule(self.c) # receives z. self.c = embedding channels = output z dim
        else:
            self.struct_module = None

        self.sig_Rl = torch.nn.Parameter(torch.tensor(10.0))

        # Handle ablation for number of key layers
        Nlayers = (
            self.ablation.num_trigon_key_layers_override 
            if self.ablation.num_trigon_key_layers_override is not None 
            else args.model_params_TR.n_trigon_key_layers
        ) if self.ablation.enable_trigon_key_layers else 0
        
        normalize = args.model_params_TR.normalize_Xform

        # Handle ligand-to-key attention ablation
        self.lig_to_key_attn = args.model_params_TR.lig_to_key_attn and self.ablation.enable_lig_to_key_attn
        
        # Handle projection layer ablations
        if not self.ablation.skip_grid_projection:
            self.grid_proj = nn.Linear(args.model_params_grid.l0_out_features, self.c)
        else:
            self.grid_proj = nn.Identity()  # Identity layer for no projection
            
        if not self.ablation.skip_key_projection:
            self.key_proj = nn.Linear(args.model_params_ligand.l0_out_features, self.c)
        else:
            self.key_proj = nn.Identity()  # Identity layer for no projection

        # Handle trigon key layers and xform modules based on ablation settings
        if self.ablation.enable_trigon_key_layers and Nlayers > 0:
            if args.model_params_TR.shared_trigon:
                trigon_key_layer = TrigonModule(n_trigonometry_module_stack=1,
                                                # 위에 projection 했으면 이거 c로 바꿔야함
                                                grid_m=self.c, #args.model_params_grid.l0_out_features,
                                                ligand_m =self.c, #args.model_params_ligand.l0_out_features, 
                                                c=self.c, dropout_rate=self.dropout_rate)
                self.trigon_key_layers = nn.ModuleList([ trigon_key_layer for _ in range(Nlayers) ])
            else:
                self.trigon_key_layers = nn.ModuleList([
                    TrigonModule(n_trigonometry_module_stack=1,
                                # 위에 projection 했으면 이거 c로 바꿔야함
                                grid_m=self.c, #args.model_params_grid.l0_out_features,
                                ligand_m =self.c, #args.model_params_ligand.l0_out_features, 
                                c=self.c, dropout_rate=self.dropout_rate) for _ in range(Nlayers) ])
        else:
            self.trigon_key_layers = nn.ModuleList([])
            
        if self.ablation.enable_xform_modules and Nlayers > 0:
            self.XformKeys = nn.ModuleList([ XformModule( self.c, normalize=normalize ) for _ in range(Nlayers) ])
            self.XformGrids = nn.ModuleList([ XformModule( self.c, normalize=normalize ) for _ in range(Nlayers) ])
        else:
            self.XformKeys = nn.ModuleList([])
            self.XformGrids = nn.ModuleList([])
        
        ## Prediction Heads
        self.class_module = ClassModule( self.c, self.c,
                                         args.model_params_aff.classification_mode,
                                         args.model_params_ligand.n_lig_global_out)


    def forward(self, Grec, Glig, keyidx, grididx, u=None,
                gradient_checkpoint=True, drop_out=False):

        # 0) if there is no grid node, return None
        Ykey_s, z_norm, aff = None, None, None
        Ggrid = Grec.subgraph( grididx )
        NullArgs = (None, None, None, None, None, None) 

        if (Ggrid.batch_num_nodes()==0).any():
            return NullArgs
        
        # 1) first process Grec to get h_rec -- "motif"-embedding
        node_features = {'0':Grec.ndata['attr'][:,:,None].float(), 'x': Grec.ndata['x'].float() }
        edge_features = {'0':Grec.edata['attr'][:,:,None].float()}

        h_rec, cs = self.GridFeaturizer(Grec, node_features, edge_features, drop_out)
        # 1-1) trim to grid part of Grec
        gridmap = torch.eye(h_rec.shape[0]).to(Grec.device)[grididx]
        h_grid = torch.matmul(gridmap, h_rec) 

        # 2) ligand embedding
        try:
            h_lig = self.LigandFeaturizer(Glig, drop_out=drop_out)
            # print(f"DEBUG: LigandFeaturizer output - h_lig shape: {h_lig.shape if h_lig is not None else 'None'}")
        except Exception as e:
            # print(f"DEBUG: LigandFeaturizer failed with error: {e}")
            return NullArgs
        
        h_lig_global = self.extract_ligand_embedding( Glig.gdata.to(Glig.device),
                                                      dropout=drop_out ) # gdata isn't std attr

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

        # 3-3) Triangle attention with all ligand atoms
        z_mask = torch.einsum('bn,bm->bnm', grid_mask, lig_mask)

        if self.ablation.enable_trigon_lig and self.trigon_lig is not None:
            z = self.trigon_lig( h_grid_batched, h_lig_batched, z_mask,
                                 D_grid, D_lig,
                                 drop_out=drop_out )
        else:
            # Initialize z with simple outer product if trigon_lig is disabled
            z = torch.einsum('bid,bjd->bijd', h_grid_batched, h_lig_batched)

        # 3-4) Ligand-> Key mapper (trim down ligand -> key)
        if self.ablation.enable_keyatom_trimming:
            Kmax = max([idx.shape[0] for idx in keyidx])
            Nmax = max([idx.shape[1] for idx in keyidx])
            key_idx = torch.zeros((Glig.batch_num_nodes().shape[0],Kmax,Nmax)).to(Grec.device) #b x K x N
            lig_to_key_mask = torch.zeros_like( key_idx ) # B x K x N
            for i,idx in enumerate(keyidx):
                key_idx[i,:idx.shape[0],:idx.shape[1]] = idx
                lig_to_key_mask[i,:idx.shape[0],:idx.shape[1]] = 1.0

            h_key_batched = torch.einsum('bkj,bjd->bkd',key_idx,h_lig_batched)
        else:
            # Use all ligand atoms as "key" atoms if trimming is disabled
            h_key_batched = h_lig_batched
            key_idx = torch.eye(h_lig_batched.shape[1]).unsqueeze(0).repeat(h_lig_batched.shape[0], 1, 1).to(Grec.device)
            lig_to_key_mask = lig_mask.unsqueeze(1).repeat(1, h_lig_batched.shape[1], 1).float()

        # 3-4) Update key features with all-atom ligand attention
        if self.lig_to_key_attn:
            # key: h_key, query: h_lig
            A = torch.einsum('bkd,bjd->bkj', h_key_batched, h_lig_batched)
            A = masked_softmax( A, lig_to_key_mask, dim=2 ) # softmax over N

            h_key_batched = h_key_batched + torch.einsum('bkj,bjd->bkd', A, h_lig_batched )
            h_key_batched = nn.functional.layer_norm(h_key_batched, h_key_batched.shape)

        # 3-5) z reshape(mask) to key atoms only
        if self.ablation.enable_keyatom_trimming:
            z = torch.einsum( 'bkj,bijd->bikd', key_idx, z)
            # 3-6) key-position-aware attn
            key_mask = torch.einsum('bkj,bj->bk', key_idx, lig_mask.float()).bool()
        else:
            # If not trimming, z already has the right dimensions and key_mask is just lig_mask
            key_mask = lig_mask
            
        z_mask = torch.einsum('bn,bm->bnm', grid_mask, key_mask)

        h_grid_batched = h_grid_batched.repeat(h_key_batched.shape[0],1,1)
        ### IF WE WANT TO USE DIFFEERNT DIMENSION for l0_out_features and c ..... 
        h_key_batched = self.key_proj(h_key_batched)
        h_grid_batched = self.grid_proj(h_grid_batched)
        ##############################################
        
        # Process trigon key layers and xform modules if enabled
        if self.ablation.enable_trigon_key_layers and len(self.trigon_key_layers) > 0:
            for trigon,xformK,xformG in zip(self.trigon_key_layers,self.XformKeys,self.XformGrids):
                # Distance transform
                if self.ablation.enable_distance_transform and self.transform_distance is not None:
                    D_key = self.transform_distance(h_key_batched)
                else:
                    D_key = torch.zeros_like(D_grid)  # Fallback if distance transform disabled

                # XForm modules if enabled
                if self.ablation.enable_xform_modules:
                    h_key_batched  = xformK( h_key_batched, h_grid_batched, z, z_mask, dim=2 ) # key/query/attn
                    h_grid_batched = xformG( h_grid_batched, h_key_batched, z, z_mask, dim=1 ) # key/query/attn
                
                # update z
                z = trigon(h_grid_batched, h_key_batched, z_mask,
                           D_grid, D_key,
                           drop_out=drop_out)
        else:
            # If trigon key layers disabled, just compute D_key for compatibility
            if self.ablation.enable_distance_transform and self.transform_distance is not None:
                D_key = self.transform_distance(h_key_batched)
            else:
                D_key = torch.zeros_like(D_grid)

        # Structure module
        if self.ablation.enable_structure_module and self.struct_module is not None:
            # Ykey_s: B x K x 3; z_norm: B x N x K x d
            # z: B x N x K x d; z_mask: B x N x K
            Ykey_s, z_norm = self.struct_module(z, z_mask, Ggrid, key_mask)
        else:
            Ykey_s, z_norm = None, None

        # 4) final prediction head
        aff = self.class_module(z, h_grid_batched, h_key_batched,
                                 lig_rep=h_lig_global, w_mask=key_mask)

        return Ykey_s, D_key, z_norm, cs, aff, None
