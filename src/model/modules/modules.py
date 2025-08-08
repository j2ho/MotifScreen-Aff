import torch
import torch.nn as nn
from src.model.utils import to_dense_batch, make_batch_vec, masked_softmax

class StructModule( nn.Module ):
    def __init__(self, m):
        super().__init__()
        self.lastlinearlayer = nn.Linear(m,1)
        self.scale = 1.0 # num_head #1.0/np.sqrt(float(d))

    def forward( self, z, z_mask, Grec, key_mask ):
        # for structureloss
        z = self.lastlinearlayer(z).squeeze(-1) #real17 ext15
        z = masked_softmax(self.scale*z, mask=z_mask, dim = 1)

        # final processing
        batchvec_rec = make_batch_vec(Grec.batch_num_nodes()).to(Grec.device)
        # r_coords_batched: B x Nmax x 3
        x = Grec.ndata['x'].squeeze()
        r_coords_batched, _ = to_dense_batch(x, batchvec_rec) # time consuming!!

        Ykey_s = torch.einsum("bij,bic->bjc",z,r_coords_batched) # "Weighted sum":  i x j , i x 3 -> j x 3
        Ykey_s = [l for l in Ykey_s]
        Ykey_s = torch.stack(Ykey_s,dim=0) # 1 x K x 3

        key_mask = key_mask[:,:,None].repeat(1,1,3).float() #b x K x 3; "which Ks are used for each b"
        Ykey_s = Ykey_s * key_mask # b x K x 3 * b x

        return Ykey_s, z #B x ? x ?


class LigandModule( nn.Module ):
    def __init__(self, dropout_rate, n_input=19, n_out=4, hidden_dim=16 ):
        super().__init__()
        self.dropoutlayer = nn.Dropout(dropout_rate)
        self.linear1 = nn.Linear(n_input, hidden_dim)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_out)

    def forward(self, ligand_features, dropout ): # 19: 10(ntors) + 3(kapp) + 1(natm) + 3(nacc/don/aro) + 3(principal)
        if dropout:
            ligand_features = self.dropoutlayer(ligand_features)
        h_lig = self.linear1(ligand_features)
        h_lig = self.layernorm(h_lig)
        h_lig = self.linear2(h_lig)
        return h_lig

class XformModule( nn.Module ):
    def __init__(self, c, normalize=False ):
        super().__init__()
        self.linear1 = nn.Linear(c, c)
        # self.layernorm = nn.LayerNorm(c)
        self.linear2 = nn.Linear(c, c)
        self.normalize = normalize

    def forward( self, V, Q, z, z_mask, dim ): # V == key & value
        # attention provided, not Q*K
        # z:  B x N x K x c
        # below assumes dim=2 (K-dimension)
        exp_z = torch.exp(z)*(z_mask.unsqueeze(-1) + 1.0e-6) # bug fix 1

        # K-summed attention on i-th N (norm-over K)
        z_denom = exp_z.sum(axis=dim).unsqueeze(dim) # B x N x 1 x c
        z = torch.div(exp_z,z_denom) # B x N x K x c #repeated over K
        z = z_mask.unsqueeze(-1)*z #bug fix 2

        Qa = self.linear1(Q) # B x K x c
        if dim == 1:
            Va = torch.einsum('bikc,bkc->bic', z, Qa) # B x N x K x c
        elif dim == 2:
            Va = torch.einsum('bikc,bic->bkc', z, Qa) # B x N x K x c

        if self.normalize:
            Va = nn.functional.layer_norm(Va, Va.shape[-2:])

        #Va = self.layernorm( V + Va )
        V = V + self.linear2(Va)

        if self.normalize:
            V = nn.functional.layer_norm(V, V.shape[-2:])

        return V

class DistanceModule(nn.Module):
    def __init__(self,d, c):
        super().__init__()
        self.linear = nn.Linear(d,c)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(c)

    def forward(self, h):
        h = self.linear(h)
        h = self.norm(self.relu(h))
        D = torch.einsum('bic,bjc->bijc', h,h ) #sort of outer product
        D = torch.softmax( D, dim = -1 )

        return D
