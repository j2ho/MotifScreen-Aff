import torch
import torch.nn as nn

class ClassModule( nn.Module ):
    def __init__(self, m, c,
                 classification_mode='ligand',
                 n_lig_emb=4 ):
        super().__init__()
        # m: originally called "embedding_channels"
        self.classification_mode = classification_mode

        if self.classification_mode == 'former':
            self.linear_z1 = nn.Linear(c, 1)
            self.map_to_L = nn.Linear( 2*c+n_lig_emb , 8 )
            self.final_linear = nn.Linear( 8, 1 )

        elif self.classification_mode.startswith('former_contrast'):
            self.linear_lig = nn.Linear(n_lig_emb, 1)
            self.Affmap = nn.Parameter( torch.rand(m) )
            self.Pcoeff = nn.Parameter( torch.Tensor([5.0]) )
            self.Poff   = nn.Parameter( torch.Tensor([-1.0]) )
            self.linear_key = nn.Parameter( torch.rand(c) )
            
            self.Gamma = nn.Parameter( torch.Tensor([0.1]) )
            
            
    def forward( self, z, hs_grid_batched, hs_key_batched,
                 lig_rep=None,
                 w_mask=None ):

        if self.classification_mode == 'former':
            att = self.linear_z1(z).squeeze(-1)
            
            att_l = torch.nn.Softmax(dim=2)(att).sum(axis=1) # b x K
            att_r = torch.nn.Softmax(dim=1)(att).sum(axis=2) # b x N

            # "attention-weighted 1-D token"
            key_rep = torch.einsum('bk,bkl -> bl', att_l, hs_key_batched)
            grid_rep = torch.einsum('bi,bil -> bl',att_r, hs_grid_batched)

            pair_rep = torch.cat([key_rep, grid_rep, lig_rep ],dim=1) # b x emb*2
            pair_rep = self.map_to_L(pair_rep) # b x L
            Aff = self.final_linear(pair_rep).squeeze(-1) # b x 1
            
        elif self.classification_mode == 'former_contrast':
            eps = 1.0e-9
            # normalized embedding across channel dim
            hs_grid_batched = torch.softmax(hs_grid_batched, axis=-1) # B x N x d
            hs_key_batched  = torch.softmax(hs_key_batched, axis=-1) # B x K x d

            # 1) contrast part
            # derive dot product to binders to be at 1.0, non-binders at 0.0
            Affmap = self.Affmap / (torch.sum(self.Affmap) + eps)# normalize so that sum be channel-dimx
            Aff_contrast = torch.einsum( 'bkd,d->bk', hs_key_batched, Affmap ) # B x k

            ## 2) per-key aff part
            # "attention-weighted 1-D probability"
            key_P = torch.einsum('bnkd,bkd -> bnk', z, hs_key_batched) #the original unnormed "z" should make more sense
            
            #max per each key across grid; range 0~10
            key_P = nn.functional.max_pool2d(key_P,kernel_size=(key_P.shape[-2],1)) # B x 1 x k 
            aff_key = self.Pcoeff*(key_P + self.Poff).squeeze(dim=1) # B x k

            # Pcoeff: 0.3, Poff: -1.8, aff_key: -0.5~0.5
            aff_key = torch.sum(aff_key*w_mask,axis=-1)/(torch.sum(w_mask,axis=-1) + 1.0e-6) # B x k -> B x 1
            aff_lig = self.linear_lig( lig_rep ).squeeze() # [-1,1]; B x 1

            Aff = ( aff_key + self.Gamma*aff_lig )/(1+self.Gamma)
                
            Aff = ( Aff, Aff_contrast )
            
        return Aff
