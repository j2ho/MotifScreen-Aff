import torch
import torch.nn as nn


class GATLayer(torch.nn.Module):
    def __init__(self, num_in_features, num_out_features,
                 num_of_heads,
                 last_layer=False,
                 add_skip_connection=True, bias=True):

        super().__init__()

        self.num_in_features = num_in_features
        self.num_of_heads = num_of_heads
        self.num_out_features = num_out_features
        self.concat = True # concat or average?
        self.add_skip_connection = add_skip_connection
        self.last_layer = last_layer

        self.linear_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

        self.layer_target = nn.Linear(num_of_heads*num_out_features, num_of_heads)
        self.layer_source = nn.Linear(num_of_heads*num_out_features, num_of_heads)

        # Bias is definitely not crucial to GAT
        self.bias = nn.Parameter( torch.zeros( num_of_heads * num_out_features ) )

        self.leakyReLU = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.activation = nn.ELU()
        self.skip_proj = nn.Linear(num_in_features, num_of_heads * num_out_features, bias=False)

    def forward(self, in_nodes_features, edge_index):
        num_of_nodes = in_nodes_features.shape[0]

        #split to head (only at 0-th layer)
        # N x c -> N x Hc
        if in_nodes_features.shape[-1] == self.num_in_features:
            nodes_features_proj = self.linear_proj(in_nodes_features)
        else:
            nodes_features_proj = in_nodes_features

        # N x Hc -> N x H
        f_source = self.layer_source(nodes_features_proj) # N x H
        f_target = self.layer_target(nodes_features_proj) # N x H

        # N -> E
        f_source = f_source.index_select(0, edge_index[0])
        f_target = f_target.index_select(0, edge_index[1])
        f_per_edge = self.activation(f_source + f_target) # E x H

        #N x Hc - > N x H x c
        nodes_features_proj = nodes_features_proj.view(-1, self.num_of_heads,self.num_out_features)
        nodes_features_proj = nodes_features_proj.index_select(0, edge_index[0]) #E x H x c

        attentions_per_edge = self.neighborhood_aware_softmax(f_per_edge, edge_index[1], num_of_nodes) # E x H x 1

        # Add stochasticity to neighborhood aggregation
        #attentions_per_edge = self.dropout(attentions_per_edge)

        # Step 3: Neighborhood aggregation
        # Element-wise (aka Hadamard) product. Operator * does the same thing as torch.mul
        # shape = (E, H, c) * (E, H, 1) -> (E, H, c), 1 gets broadcast into FOUT
        nodes_features_proj_weighted = nodes_features_proj * attentions_per_edge

        # This part sums up weighted and projected neighborhood feature vectors for every target node
        # aggregate E -> N, H, c
        size = list(nodes_features_proj_weighted.shape)
        size[0] = num_of_nodes  # shape = (N, H, c)

        # (E) -> (E, H, c)
        trg_index_broadcasted = self.explicit_broadcast(edge_index[1], nodes_features_proj_weighted)

        # (E, H, c) -> (N, H, c)
        out_nodes_features = torch.zeros(size, dtype=in_nodes_features.dtype, device=in_nodes_features.device)
        out_nodes_features.scatter_add_(0, trg_index_broadcasted, nodes_features_proj_weighted)
        # Step 4: Residual/skip connections, concat and bias
        if self.last_layer:
            out_nodes_features = out_nodes_features.mean(dim=1) #mean over heads: N x c
        else:
            out_nodes_features = self.skip_concat_bias(attentions_per_edge, in_nodes_features, out_nodes_features) # N x Hc

        return out_nodes_features #, edge_index

    def skip_concat_bias(self, attention_coefficients, in_nodes_features, out_nodes_features):
        if not out_nodes_features.is_contiguous():
            out_nodes_features = out_nodes_features.contiguous()

        out_nodes_features += self.skip_proj(in_nodes_features).view(-1, self.num_of_heads, self.num_out_features)
        out_nodes_features = out_nodes_features.view(-1, self.num_of_heads * self.num_out_features)

        out_nodes_features += self.bias

        return self.activation(out_nodes_features)

    def neighborhood_aware_softmax(self, f_per_edge, trg_index, num_of_nodes):
        f_per_edge = f_per_edge - f_per_edge.max()
        exp_f_per_edge = f_per_edge.exp()  # softmax

        # Calculate the denominator. shape = (E, NH)
        neigborhood_aware_denominator = self.sum_edge_f_neighborhood_aware(exp_f_per_edge, trg_index, num_of_nodes)

        # E x H x c; E x N x c
        attentions_per_edge = exp_f_per_edge / (neigborhood_aware_denominator + 1e-10)

        return attentions_per_edge.unsqueeze(-1)

    def sum_edge_f_neighborhood_aware(self, exp_f_per_edge, trg_index, num_of_nodes):
        # num edge
        # E -> E x h x c
        trg_index_broadcasted = self.explicit_broadcast(trg_index, exp_f_per_edge)

        # shape = (N, NH), where N is the number of nodes and NH the number of attention heads
        size = list(exp_f_per_edge.shape)  # convert to list otherwise assignment is not possible
        size[0] = num_of_nodes
        neighborhood_sums = torch.zeros(size, dtype=exp_f_per_edge.dtype, device=exp_f_per_edge.device)

        neighborhood_sums.scatter_add_(0, trg_index_broadcasted, exp_f_per_edge)
        return neighborhood_sums.index_select(0, trg_index)

    def explicit_broadcast(self, this, other):
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)