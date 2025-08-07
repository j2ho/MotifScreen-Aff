import torch
from torch import nn
from torch.nn import Linear
from torch.utils import checkpoint


class TriangleProteinToCompound(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        self.gate_linear1 = Linear(embedding_channels, c)
        self.gate_linear2 = Linear(embedding_channels, c)

        self.linear1 = Linear(embedding_channels, c)
        self.linear2 = Linear(embedding_channels, c)
        self.linear3 = Linear(c,c)
        self.linear4 = Linear(c,c)

        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
        
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        # z_mask : torch.Size([b, i, j, 1])
        z = self.layernorm(z)

        protein_pair = self.layernorm(protein_pair)
        compound_pair = self.layernorm(compound_pair)
 
        ab1 = self.gate_linear1(z).sigmoid() * self.linear1(z) * z_mask #torch.Size([b, i, j, c])
        ab2 = self.gate_linear2(z).sigmoid() * self.linear2(z) * z_mask

        protein_pair = self.gate_linear2(protein_pair).sigmoid() * self.linear2(protein_pair)
        compound_pair = self.gate_linear1(compound_pair).sigmoid() * self.linear1(compound_pair)

        g = self.ending_gate_linear(z).sigmoid()

        size_p = protein_pair.shape 
        size_c = compound_pair.shape 

        block1 = torch.einsum("bikc,bkjc->bijc", protein_pair, ab1)
        block2 = torch.einsum("bikc,bjkc->bijc", ab2, compound_pair)

        z = g * self.linear_after_sum(self.layernorm_c(block1+block2)) * z_mask
        return z


class Self_Attention(nn.Module):
    def __init__(self, hidden_size,num_attention_heads=8,drop_rate=0.5):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dp = nn.Dropout(drop_rate)
        self.ln = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,q,k,v,attention_mask=None,attention_weight=None):
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        attention_scores = torch.matmul(q, k.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # attention_probs = self.dp(attention_probs)
        if attention_weight is not None:
            attention_weight_sorted_sorted = torch.argsort(torch.argsort(-attention_weight,axis=-1),axis=-1)
            # if self.training:
            #     top_mask = (attention_weight_sorted_sorted<np.random.randint(28,45))
            # else:
            top_mask = (attention_weight_sorted_sorted<32)
            attention_probs = attention_probs * top_mask
            # attention_probs = attention_probs * attention_weight
            attention_probs = attention_probs / (torch.sum(attention_probs,dim=-1,keepdim=True) + 1e-5)
        # print(attention_probs.shape,v.shape)
        # attention_probs = self.dp(attention_probs)
        outputs = torch.matmul(attention_probs, v)

        outputs = outputs.permute(0, 2, 1, 3).contiguous()
        new_output_shape = outputs.size()[:-2] + (self.all_head_size,)
        outputs = outputs.view(*new_output_shape)
        outputs = self.ln(outputs)
        return outputs


class TriangleSelfAttentionRowWise(torch.nn.Module):
    # use the protein-compound matrix only.
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # self.dp = nn.Dropout(drop_rate)
        # self.ln = nn.LayerNorm(hidden_size)

        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        # self.layernorm_c = torch.nn.LayerNorm(c)

        self.linear_q = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_k = Linear(embedding_channels, self.all_head_size, bias=False)
        self.linear_v = Linear(embedding_channels, self.all_head_size, bias=False)
        # self.b = Linear(embedding_channels, h, bias=False)
        self.g = Linear(embedding_channels, self.all_head_size)
        self.final_linear = Linear(self.all_head_size, embedding_channels)

    def reshape_last_dim(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, z, z_mask):
        # z[b,i,j,:] = interaction features between grid point i and ligand atom j in batch b
        z = self.layernorm(z)
        
        # Check for NaN after layernorm
        if torch.isnan(z).any():
            print("WARNING: NaN detected in TriangleSelfAttentionRowWise after layernorm")
            z = torch.nan_to_num(z, nan=0.0, posinf=1e6, neginf=-1e6)
        
        p_length = z.shape[1]
        batch_n = z.shape[0]
        z_i = z
        z_mask_i = z_mask.view((batch_n, p_length, 1, 1, -1))
        attention_mask_i = (1e9 * (z_mask_i.float() - 1.))
        q = self.reshape_last_dim(self.linear_q(z_i)) 
        k = self.reshape_last_dim(self.linear_k(z_i))
        v = self.reshape_last_dim(self.linear_v(z_i))
        
        # h = self.all_head_size, c = attention_head_size
        logits = torch.einsum('biqhc,bikhc->bihqk', q, k) + attention_mask_i
        
        # Check for NaN in attention logits and add numerical stability
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            print("WARNING: NaN/Inf detected in attention logits")
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Add attention scaling for numerical stability
        scale = 1.0 / (self.attention_head_size ** 0.5)
        logits = logits * scale
        
        weights = nn.Softmax(dim=-1)(logits)
        
        # Check attention weights
        if torch.isnan(weights).any():
            print("WARNING: NaN detected in attention weights")
            weights = torch.nan_to_num(weights, nan=0.0)
        
        weighted_avg = torch.einsum('bihqk,bikhc->biqhc', weights, v)
        g = self.reshape_last_dim(self.g(z_i)).sigmoid()
        output = g * weighted_avg
        new_output_shape = output.size()[:-2] + (self.all_head_size,)
        output = output.view(*new_output_shape)
        z = output
        z = self.final_linear(z) * z_mask.unsqueeze(-1)
        
        # Final NaN check
        if torch.isnan(z).any():
            print("WARNING: NaN detected in TriangleSelfAttentionRowWise output")
            z = torch.nan_to_num(z, nan=0.0)
        
        return z


class Transition(torch.nn.Module):
    # separate left/right edges (block1/block2).
    def __init__(self, embedding_channels=256, n=4, bias=True):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n*embedding_channels, bias=bias)
        self.linear2 = Linear(n*embedding_channels, embedding_channels, bias=bias)
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        z = self.layernorm(z)
        z = self.linear2((self.linear1(z)).relu())
        return z
    
    
class TrigonModule(nn.Module):
    def __init__(self,
                 n_trigonometry_module_stack,
                 grid_m=64,
                 ligand_m=64, 
                 c=64,
                 dropout_rate=0.1,
                 bias=True,
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.n_trigonometry_module_stack = n_trigonometry_module_stack

        self.Wrs = nn.Linear(grid_m,c,bias=bias)
        self.Wls = nn.Linear(ligand_m,c,bias=bias)

        self.protein_to_compound_list = nn.ModuleList([TriangleProteinToCompound(embedding_channels=c, c=c) for _ in range(n_trigonometry_module_stack)])
        self.triangle_self_attention_list = nn.ModuleList([TriangleSelfAttentionRowWise(embedding_channels=c, c=c) for _ in range(n_trigonometry_module_stack)])

        self.transition = Transition(embedding_channels=c, n=4, bias=bias)

    def forward(self, hs_rec, hs_lig, z_mask,
                D_rec, D_lig,
                use_checkpoint=False, drop_out=False):
        # hs_rec: B x Nmax x d
        # hs_lig: B x Mmax x d

        # Check inputs for NaN
        if torch.isnan(hs_rec).any():
            print("WARNING: NaN detected in TrigonModule input hs_rec")
            hs_rec = torch.nan_to_num(hs_rec, nan=0.0)
        if torch.isnan(hs_lig).any():
            print("WARNING: NaN detected in TrigonModule input hs_lig")
            hs_lig = torch.nan_to_num(hs_lig, nan=0.0)

        # process features
        hs_rec = self.Wrs(hs_rec)
        hs_lig = self.Wls(hs_lig)
        
        # Check after linear projections
        if torch.isnan(hs_rec).any() or torch.isnan(hs_lig).any():
            print("WARNING: NaN detected after TrigonModule linear projections")
            hs_rec = torch.nan_to_num(hs_rec, nan=0.0)
            hs_lig = torch.nan_to_num(hs_lig, nan=0.0)
        
        # initial pairwise features
        z = torch.einsum('bnd,bmd->bnmd', hs_rec, hs_lig )
        
        # Check initial z
        if torch.isnan(z).any():
            print("WARNING: NaN detected in initial TrigonModule pairwise features")
            z = torch.nan_to_num(z, nan=0.0)

        # trigonometry part
        for i_module in range(self.n_trigonometry_module_stack):
            if use_checkpoint:
                # distance aware cross attention
                zadd = checkpoint.checkpoint(self.protein_to_compound_list[i_module], z, D_rec, D_lig, z_mask.unsqueeze(-1))
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                # triangle self attention
                zadd = checkpoint.checkpoint(self.triangle_self_attention_list[i_module], z, z_mask)
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
            else:
                # distance aware cross attention
                zadd = self.protein_to_compound_list[i_module](z, D_rec, D_lig, z_mask.unsqueeze(-1))
                if drop_out: zadd = self.dropout(zadd)
                
                # Check zadd for NaN
                if torch.isnan(zadd).any():
                    print(f"WARNING: NaN detected in TrigonModule cross-attention layer {i_module}")
                    zadd = torch.nan_to_num(zadd, nan=0.0)
                
                z = z + zadd
                
                # triangle self attention
                zadd = self.triangle_self_attention_list[i_module](z, z_mask)
                if drop_out: zadd = self.dropout(zadd)
                
                # Check zadd for NaN
                if torch.isnan(zadd).any():
                    print(f"WARNING: NaN detected in TrigonModule self-attention layer {i_module}")
                    zadd = torch.nan_to_num(zadd, nan=0.0)
                
                z = z + zadd

            # Check z before transition
            if torch.isnan(z).any():
                print(f"WARNING: NaN detected in TrigonModule before transition layer {i_module}")
                z = torch.nan_to_num(z, nan=0.0)

            # norm -> linear -> relu -> linear
            z = self.transition(z)
            
            # Check z after transition
            if torch.isnan(z).any():
                print(f"WARNING: NaN detected in TrigonModule after transition layer {i_module}")
                z = torch.nan_to_num(z, nan=0.0)

        return z
