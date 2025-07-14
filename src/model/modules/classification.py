import torch
import torch.nn as nn
from src.model.utils import masked_softmax


class ClassModule( nn.Module ):
    def __init__(self, m, c,
                 classification_mode='ligand',
                 n_lig_emb=4 ):
        super().__init__()
        # m: originally called "embedding_channels"
        self.classification_mode = classification_mode
        
        if self.classification_mode == 'ligand':
            self.lapool = nn.AdaptiveMaxPool2d((4,c)) #nn.Linear(l0_out_features,c)
            self.rapool = nn.AdaptiveMaxPool2d((100,c))
            self.lhpool = nn.AdaptiveMaxPool2d((20,c))
            self.linear_pre_aff = nn.Linear(c,1)
            self.linear_for_aff = nn.Linear(124,1)
            
        elif self.classification_mode in ['ligand_v2','ligand_v3','combo_v1']:
            self.wra = nn.Linear(m, m) #, bias=None)
            self.wrh = nn.Linear(m, m) #, bias=None)
            self.wla = nn.Linear(m, m)#, bias=None)
            self.wlh = nn.Linear(m, m)#, bias=None)
            self.linear_z1 = nn.Linear(m,1)
            M = {'ligand_v2':2*m,'ligand_v3':2*m+n_lig_emb,'combo_v1':2*m+n_lig_emb}[self.classification_mode]
            L = {'ligand_v2':3,'ligand_v3':5,'combo_v1':5}[self.classification_mode]

            if self.classification_mode == 'combo_v1':
                self.w_cR = nn.Linear( m, m )
                self.w_cl = nn.Linear( m, m )
                self.linear_kR = nn.Linear( m, m )
            
            self.map_to_L = nn.Linear(M,L)
            self.final_linear = nn.Linear(L, 1)

        elif self.classification_mode == 'former':
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
            
        elif self.classification_mode == 'tank':
            # attention matrix to affinity 
            self.linear1 = nn.Linear(m, 1)
            self.linear2 = nn.Linear(m, 1)
            self.linear1a = nn.Linear(m, m)
            self.linear2a = nn.Linear(m, m)
            self.bias = nn.Parameter(torch.ones(1))
            self.leaky = nn.LeakyReLU()
            
    def forward( self, z, hs_grid_batched, hs_key_batched,
                 lig_rep=None,
                 hs_rec_batched=None,
                 w_Rl=None, w_mask=None ):

        ## TODO
        # for classification

            
        if self.classification_mode == 'former':
            ## simplified!
            att = self.linear_z1(z).squeeze(-1)
            #z = torch.einsum('bnh,bkh->bnk', hs_grid_batched, hs_key_batched) # b x N x k
            
            att_l = torch.nn.Softmax(dim=2)(att).sum(axis=1) # b x K
            att_r = torch.nn.Softmax(dim=1)(att).sum(axis=2) # b x N

            # "attention-weighted 1-D token"
            key_rep = torch.einsum('bk,bkl -> bl', att_l, hs_key_batched)
            grid_rep = torch.einsum('bi,bil -> bl',att_r, hs_grid_batched)

            #
            pair_rep = torch.cat([key_rep, grid_rep, lig_rep ],dim=1) # b x emb*2
            pair_rep = self.map_to_L(pair_rep) # b x L
            Aff = self.final_linear(pair_rep).squeeze(-1) # b x 1
            
        elif self.classification_mode in ['former_contrast','former_contrast2']:
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
            # opt1
            key_P = torch.einsum('bnkd,bkd -> bnk', z, hs_key_batched) #the original unnormed "z" should make more sense
            
            #max per each key across grid; range 0~10
            key_P = nn.functional.max_pool2d(key_P,kernel_size=(key_P.shape[-2],1)) # B x 1 x k 
            #if self.classification_mode == 'former_contrast3':
            #    Aff = torch.sigmoid(torch.mean(key_P,axis=(1,2)))
            #else:
            aff_key = self.Pcoeff*(key_P + self.Poff).squeeze(dim=1) # B x k

            # Pcoeff: 0.3, Poff: -1.8, aff_key: -0.5~0.5
            aff_key = torch.sum(aff_key*w_mask,axis=-1)/(torch.sum(w_mask,axis=-1) + 1.0e-6) # B x k -> B x 1
            aff_lig = self.linear_lig( lig_rep ).squeeze() # [-1,1]; B x 1

            Aff = ( aff_key + self.Gamma*aff_lig )/(1+self.Gamma)
                
            Aff = ( Aff, Aff_contrast )
            
        # tested but forbidden?
        elif self.classification_mode in ['former_contrast3']:
            #hs_key_batched: layernormed
            
            eps = 1.0e-9
            # 1) contrast part
            # derive dot product to binders to be at 1.0, non-binders at 0.0

            # measure just direction
            hs_key_normed = hs_key_batched / hs_key_batched.norm(dim=-1).unsqueeze(-1)
            Aff_contrast = torch.einsum( 'bkd,d->bk', hs_key_normed, self.Affmap ) # B x k

            ## 2) per-key aff part
            # simplest -- measure "per-key-score"
            # why hs_key_batch becomes < 0?
            aff_key = torch.einsum( 'bkd,d->bk', hs_key_batched, self.linear_key ).unsqueeze(1) # B x 1 x k
            #print(hs_key_batched, aff_key)
            
            # sum over keys; individual elem can be neg or pos
            Aff = torch.mean(aff_key*w_mask,axis=(1,2)) #
            #Aff = torch.sigmoid(Aff) #Aff should be logit, not probability

            Aff = ( Aff, Aff_contrast ) 
            
        else:
            exp_z = torch.exp(z) 
            # soft alignment 
            # normalize each row of z for receptor counterpart
            zr_denom = exp_z.sum(axis=(-2)).unsqueeze(-2) # 1 x Nrec x 1 x c
            zr = torch.div(exp_z,zr_denom) # 1 x Nrec x K x c; "per-NK weight, receptor version"
            ra = zr*hs_key_batched.unsqueeze(1) # 1 x Nrec x K x c
            ra = ra.sum(axis=-2) # 1 x Nrec x c
            
            # normalize each row of z for ligand counterpart
            zl_denom = exp_z.sum(axis=(-3)).unsqueeze(-3) # 1 x Nrec x 1 x c
            zl = torch.div(exp_z,zl_denom) # 1 x Nrec x K x c; "per-NK weight, ligand version"
            zl_t = torch.transpose(zl, 1, 2) # 1 x K x Nrec x c

            la = zl_t*hs_grid_batched.unsqueeze(1) # 1 x K x Nrec x numchannel
            la = la.sum(axis=-2) # 1 x K x numchannel
            
            if self.classification_mode == 'ligand':
                # concat and then pool 
                la = self.lapool(la) # 1 x K x c
                lh = hs_key_batched # 1 x Nlig x c
                ra = self.rapool(ra) # 1 x 100 x c
                lh = self.lhpool(lh) # 1 x 20 x c

                cat = torch.cat([ra,la,lh],dim=1) # 1 x 124 x c
                
                Aff = self.linear_pre_aff(cat).squeeze(-1) # 1 x 124
                Aff = self.linear_for_aff(Aff).squeeze(-1) # b x 1

            elif self.classification_mode in ['ligand_v2', 'ligand_v3', 'combo_v1']:
                ra_rh = (self.wra(ra) + self.wrh(hs_grid_batched))
                la_lh = (self.wla(la) + self.wlh(hs_key_batched)) # b x K x emb

                att = (self.linear_z1(z)).squeeze(-1) # b x Ngrid x K
                att_l = torch.nn.Softmax(dim=2)(att).sum(axis=1) # b x K
                att_r = torch.nn.Softmax(dim=1)(att).sum(axis=2) # b x Ngrid

                key_rep = torch.einsum('bk,bkl -> bl', att_l, la_lh)
                grid_rep = torch.einsum('bk,bkl -> bl',att_r, ra_rh)

                if self.classification_mode == 'combo_v1':
                    Rh = self.w_cR( hs_rec_batched ) #1 x N x h; dimension preseved
                    lh = self.w_cl( hs_key_batched ) #b x K x h

                    z_Rl = torch.einsum('bnh,bkh->bnk', Rh, lh) # b x n x k 
                    w_mask = w_mask[:,None,:].repeat((1,z_Rl.shape[1],1))
                    z_Rl = masked_softmax(w_Rl*z_Rl, mask=w_mask, dim=1)

                    # actually bnk,1nh -> bkh: expansion to B
                    key_rep_from_R = torch.einsum( 'bnk,bnh -> bkh', z_Rl, hs_rec_batched )
                    key_rep_from_R = self.linear_kR( key_rep_from_R ) # b x k x h
                    # reuse att_l
                    key_rep_from_R = torch.einsum('bk,bkh -> bh', att_l, key_rep_from_R)
                    
                    key_rep = key_rep + key_rep_from_R
                    pair_rep = torch.cat([key_rep, lig_rep, grid_rep],dim=1) # b x emb*2 + L
                    
                elif self.classification_mode == 'ligand_v2':
                    pair_rep = torch.cat([key_rep, grid_rep],dim=1) # b x emb*2
                elif self.classification_mode == 'ligand_v3':
                    pair_rep = torch.cat([key_rep, lig_rep, grid_rep],dim=1) # b x emb*2 + L
                    
                pair_rep = self.map_to_L(pair_rep) # b x L
                Aff = self.final_linear(pair_rep).squeeze(-1) # b x 1
            
        return Aff
