import torch
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F


class Transition(nn.Module):
    def __init__(self, embedding_channels=256, n=4, bias=True):
        super().__init__()
        self.layernorm = nn.LayerNorm(embedding_channels)
        self.linear1 = Linear(embedding_channels, n*embedding_channels, bias=bias)
        self.linear2 = Linear(n*embedding_channels, embedding_channels, bias=bias)
    
    def forward(self, z):
        z = self.layernorm(z)
        z = self.linear2(F.relu(self.linear1(z)))
        return z

class TriangleProteinToCompound(torch.nn.Module):
    def __init__(self, embedding_channels=256, c=128):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(embedding_channels)
        self.layernorm_c = torch.nn.LayerNorm(c)

        self.gate_linear1 = Linear(embedding_channels, c)
        self.gate_linear2 = Linear(embedding_channels, c)
        self.linear1 = Linear(embedding_channels, c)
        self.linear2 = Linear(embedding_channels, c)

        self.ending_gate_linear = Linear(embedding_channels, embedding_channels)
        self.linear_after_sum = Linear(c, embedding_channels)
        
    def forward(self, z, protein_pair, compound_pair, z_mask):
        # Input shapes: z[5,861,36,64], protein_pair[1,861,861,64], compound_pair[5,36,36,64]
        
        # Minimize layernorm calls
        z_norm = self.layernorm(z)
        protein_norm = self.layernorm(protein_pair) 
        compound_norm = self.layernorm(compound_pair)
        
        # Instead of: sigmoid(gate(x)) * linear(x) * mask, compute gate and linear, then fuse sigmoid*linear*mask
        gate1_out = self.gate_linear1(z_norm)
        linear1_out = self.linear1(z_norm)
        ab1 = torch.sigmoid(gate1_out) * linear1_out * z_mask
        
        gate2_out = self.gate_linear2(z_norm) 
        linear2_out = self.linear2(z_norm)
        ab2 = torch.sigmoid(gate2_out) * linear2_out * z_mask

        # Process protein and compound pairs
        protein_gate = self.gate_linear2(protein_norm)
        protein_linear = self.linear2(protein_norm)
        protein_processed = torch.sigmoid(protein_gate) * protein_linear
        
        compound_gate = self.gate_linear1(compound_norm)
        compound_linear = self.linear1(compound_norm)
        compound_processed = torch.sigmoid(compound_gate) * compound_linear

        # Instead of expanding protein_pair to [5,861,861,64] (uses 5x memory),...  
        if protein_processed.size(0) == 1 and ab1.size(0) > 1:
            #  squeeze batch dim, use optimized einsum, then unsqueeze = better? 
            protein_squeezed = protein_processed.squeeze(0)  # [861,861,64]
            
            # 'ikc,bkjc->bijc' is more efficient than 'bikc,bkjc->bijc' with broadcasting ? 
            block1 = torch.einsum('ikc,bkjc->bijc', protein_squeezed, ab1)
        else:
            block1 = torch.einsum('bikc,bkjc->bijc', protein_processed, ab1)
        
        block2 = torch.einsum('bikc,bjkc->bijc', ab2, compound_processed)
        
        g = torch.sigmoid(self.ending_gate_linear(z_norm))
        
        # Combine add + layernorm + linear + multiply in sequence to minimize intermediate tensors
        sum_result = block1 + block2
        normalized = self.layernorm_c(sum_result)  
        final_linear = self.linear_after_sum(normalized)
        result = g * final_linear * z_mask
        
        return result


class TriangleSelfAttention(torch.nn.Module):
    def __init__(self, embedding_channels=128, c=32, num_attention_heads=4):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = c
        self.all_head_size = num_attention_heads * c
        
        self.layernorm = nn.LayerNorm(embedding_channels)
        
        # Single QKV projection to reduce linear layer overhead
        self.qkv_proj = Linear(embedding_channels, 3 * self.all_head_size, bias=False)
        self.gate_proj = Linear(embedding_channels, self.all_head_size)
        self.out_proj = Linear(self.all_head_size, embedding_channels)
        
        # Pre-compute scale factor
        self.scale = 1.0 / (c ** 0.5)

    def forward(self, z, z_mask):
        # z: [5, 861, 36, 64], z_mask: [5, 861, 36]
        batch_size, seq_len_i, seq_len_j, embed_dim = z.shape
        
        z_norm = self.layernorm(z)
        
        # single projection instead of 3 
        qkv = self.qkv_proj(z_norm)  # [5, 861, 36, 3*all_head_size]
        
        # Reshape for multi-head attention
        qkv = qkv.view(batch_size, seq_len_i, seq_len_j, 3, self.num_attention_heads, self.attention_head_size)
        q, k, v = qkv.unbind(dim=3)  # Each: [5, 861, 36, num_heads, head_dim]
        
        # Pre-compute and cache mask expansion
        mask_expanded = z_mask.unsqueeze(-2).unsqueeze(-1)  # [5, 861, 1, 36, 1] 
        mask_full = mask_expanded.expand(-1, -1, self.num_attention_heads, -1, seq_len_j)  # [5, 861, H, 36, 36]
        attention_mask = torch.where(mask_full, 0.0, -1e9)
        
        scores = torch.einsum('biqhc,bikhc->bihqk', q, k) * self.scale
        scores = scores + attention_mask
        
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.einsum('bihqk,bikhc->biqhc', weights, v)
        
        g = self.gate_proj(z_norm).view(batch_size, seq_len_i, seq_len_j, self.num_attention_heads, self.attention_head_size)
        attn_out = torch.sigmoid(g) * attn_out
        
        attn_out = attn_out.view(batch_size, seq_len_i, seq_len_j, self.all_head_size)
        output = self.out_proj(attn_out) * z_mask.unsqueeze(-1)
        
        return output


class TrigonModule(nn.Module):
    def __init__(self,
                 n_trigonometry_module_stack,
                 grid_m=64,
                 ligand_m=64, 
                 c=64,
                 dropout_rate=0.1,
                 bias=True):
        super().__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.n_trigonometry_module_stack = n_trigonometry_module_stack

        self.Wrs = nn.Linear(grid_m, c, bias=bias)
        self.Wls = nn.Linear(ligand_m, c, bias=bias)

        self.protein_to_compound_list = nn.ModuleList([
            TriangleProteinToCompound(embedding_channels=c, c=c) 
            for _ in range(n_trigonometry_module_stack)
        ])
        
        self.triangle_self_attention_list = nn.ModuleList([
            TriangleSelfAttention(embedding_channels=c, c=c) 
            for _ in range(n_trigonometry_module_stack)
        ])

        self.transition = Transition(embedding_channels=c, n=4, bias=bias)

    def forward(self, hs_rec, hs_lig, z_mask, D_rec, D_lig, use_checkpoint=False, drop_out=False):
        # Pre-compute features
        hs_rec = self.Wrs(hs_rec)
        hs_lig = self.Wls(hs_lig)
        
        # Initial pairwise features - this einsum is already optimal
        z = torch.einsum('bnd,bmd->bnmd', hs_rec, hs_lig)

        # Main processing loop
        for i_module in range(self.n_trigonometry_module_stack):
            if use_checkpoint:
                zadd = torch.utils.checkpoint.checkpoint(
                    self.protein_to_compound_list[i_module], 
                    z, D_rec, D_lig, z_mask.unsqueeze(-1),
                    use_reentrant=False
                )
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                
                zadd = torch.utils.checkpoint.checkpoint(
                    self.triangle_self_attention_list[i_module], 
                    z, z_mask,
                    use_reentrant=False
                )
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
            else:
                zadd = self.protein_to_compound_list[i_module](z, D_rec, D_lig, z_mask.unsqueeze(-1))
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd
                
                zadd = self.triangle_self_attention_list[i_module](z, z_mask)
                if drop_out: zadd = self.dropout(zadd)
                z = z + zadd

            z = self.transition(z)

        return z
