import torch
import torch.nn as nn

class AbsAffModule(nn.Module):
    def __init__(self, m, c,
                 classification_module=None,
                 n_lig_emb=4,
                 hidden_dim=64):
        """
        Initialize the Absolute Affinity Module with shared parameters

        Args:
            m: Dimension of feature embeddings (grid/key)
            c: Channel dimension
            classification_module: Existing ClassModule to share parameters with
            classification_mode: Mode for classification
            n_lig_emb: Dimension of ligand global representation
            hidden_dim: Hidden dimension for regression layer
        """
        super().__init__()

        # Share parameters with classification module if provided
        if classification_module is not None:
            self.Affmap = classification_module.Affmap
            self.Pcoeff = classification_module.Pcoeff
            self.Poff = classification_module.Poff
            self.linear_key = classification_module.linear_key
            self.Gamma = classification_module.Gamma
            self.linear_lig = classification_module.linear_lig
        else:
            # Otherwise create new parameters
            self.Affmap = nn.Parameter(torch.rand(m))
            self.Pcoeff = nn.Parameter(torch.Tensor([5.0]))
            self.Poff = nn.Parameter(torch.Tensor([-1.0]))
            self.linear_key = nn.Parameter(torch.rand(c))
            self.Gamma = nn.Parameter(torch.Tensor([0.1]))
            self.linear_lig = nn.Linear(n_lig_emb, 1)

        # Additional layers specific to absolute affinity prediction
        self.regression_layer = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

        self.min_affinity = nn.Parameter(torch.Tensor([1.0]))
        self.max_affinity = nn.Parameter(torch.Tensor([15.0]))

    def forward(self, z, hs_grid_batched, hs_key_batched,
                lig_rep=None,
                w_mask=None):
        """
        Forward pass for absolute affinity prediction
        """
        eps = 1.0e-9

        # Same normalization as in classification
        hs_grid_batched = torch.softmax(hs_grid_batched, axis=-1)  # B x N x d
        hs_key_batched = torch.softmax(hs_key_batched, axis=-1)    # B x K x d

        # 1) Contrast part - same as classification
        Affmap = self.Affmap / (torch.sum(self.Affmap) + eps)
        Aff_contrast = torch.einsum('bkd,d->bk', hs_key_batched, Affmap)  # B x k

        # 2) Per-key affinity part - same as classification
        key_P = torch.einsum('bnkd,bkd->bnk', z, hs_key_batched)
        key_P = nn.functional.max_pool2d(key_P, kernel_size=(key_P.shape[-2], 1))  # B x 1 x k
        aff_key = self.Pcoeff * (key_P + self.Poff).squeeze(dim=1)  # B x k

        # Handle variable key atoms with masking
        if w_mask is not None:
            aff_key_avg = torch.sum(aff_key * w_mask, axis=-1) / (torch.sum(w_mask, axis=-1) + eps)
            aff_contrast_avg = torch.sum(Aff_contrast * w_mask, axis=-1) / (torch.sum(w_mask, axis=-1) + eps)
        else:
            aff_key_avg = torch.mean(aff_key, axis=-1)
            aff_contrast_avg = torch.mean(Aff_contrast, axis=-1)

        # Process ligand representation same as classification
        aff_lig = self.linear_lig(lig_rep).squeeze() if lig_rep is not None else 0

        # Calculate classification-like score
        class_score = (aff_key_avg + self.Gamma * aff_lig) / (1 + self.Gamma)

        # Stack the two scores for regression
        combined_features = torch.stack([class_score, aff_contrast_avg], dim=1)

        # Final regression to absolute affinity value
        raw_output = self.regression_layer(combined_features)

        # Convert to appropriate scale for binding affinity (e.g., pKd)
        # Typical pKd values range from 3 (weak) to 12 (strong)
        pAff = self.min_affinity + (self.max_affinity - self.min_affinity) * torch.sigmoid(raw_output)

        return pAff


class ClassAffModule(nn.Module):
    def __init__(self, m, c, n_lig_emb=4):
        super().__init__()
        # Parameters for classification part (preserved from ClassModule)
        self.linear_lig_class = nn.Linear(n_lig_emb, 1)
        self.Affmap = nn.Parameter(torch.rand(m))
        self.Pcoeff = nn.Parameter(torch.Tensor([5.0]))
        self.Poff = nn.Parameter(torch.Tensor([-1.0]))
        self.linear_key = nn.Parameter(torch.rand(c))
        self.Gamma = nn.Parameter(torch.Tensor([0.1]))
        
        # Parameters for affinity prediction
        self.z_attn = nn.Parameter(torch.rand(c))  # For weighted pooling of z features
        self.z_proj = nn.Linear(c,16)
        self.lig_proj = nn.Linear(n_lig_emb, 16)  # Project global ligand features
 
        # Final layers for affinity prediction
        self.aff_combine = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Scale parameters for pKa range
        self.pka_min = nn.Parameter(torch.Tensor([1.0]))
        self.pka_max = nn.Parameter(torch.Tensor([15.0]))
        
    def forward(self, z, hs_key_batched, lig_rep=None, w_mask=None, z_mask=None):
        # Classification part (as in original ClassModule)
        eps = 1.0e-9
        # hs_grid_batched_norm = torch.softmax(hs_grid_batched, axis=-1)
        hs_key_batched_norm = torch.softmax(hs_key_batched, axis=-1)
        
        # Contrast part
        Affmap_norm = self.Affmap / (torch.sum(self.Affmap) + eps)
        Aff_contrast = torch.einsum('bkd,d->bk', hs_key_batched_norm, Affmap_norm)
        # Per-key affinity part
        key_P = torch.einsum('bnkd,bkd->bnk', z, hs_key_batched_norm)
        key_P = nn.functional.max_pool2d(key_P, kernel_size=(key_P.shape[-2], 1))
        aff_key = self.Pcoeff * (key_P + self.Poff).squeeze(dim=1)
        aff_key = torch.sum(aff_key * w_mask, axis=-1) / (torch.sum(w_mask, axis=-1) + 1.0e-6)
        
        aff_lig_class = self.linear_lig_class(lig_rep).squeeze()
        gamma = self.Gamma
        Aff = (aff_key + gamma * aff_lig_class) / (1 + gamma.detach())
        
        # Affinity prediction using z
        # First, pool z across grid and key dimensions with attention weighting
        z_attn_norm = nn.functional.softmax(self.z_attn, dim=0)  # Normalize weights
        # Weight z features along feature dimension
        z_attn = z * z_attn_norm  # Shape: B x N x K x d
        
        # Find maximum interaction score for each key atom
        z_weighted = torch.sum(z_attn, dim=-1)  # Shape: B x N x K
        z_weighted_masked = z_weighted * z_mask  # Shape: B x N x K
        z_weights = z_weighted_masked / (torch.sum(z_weighted_masked, dim=(1, 2), keepdim=True) + eps)
        z_weighted = z * z_weights.unsqueeze(-1)  # Shape: B x N x K x d
        z_pooled = torch.sum(z_weighted, dim=(1, 2))  # Shape: B x d
        z_features = self.z_proj(z_pooled)  # Shape: B x 16

        # Project ligand global features
        lig_features = self.lig_proj(lig_rep)  # Shape: B x 16
        # Combine features
        combined = torch.cat([z_features, lig_features], dim=1)  # Shape: B x 32
        
        # Predict raw affinity
        raw_aff = self.aff_combine(combined).squeeze()  # Shape: B
        # Scale to pKa range
        pAff = self.pka_min + torch.sigmoid(raw_aff) * (self.pka_max - self.pka_min)
        return (Aff, Aff_contrast), pAff