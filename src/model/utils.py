import torch
from typing import Optional, Tuple
from torch import Tensor
import logging # For logging warnings/errors
logger = logging.getLogger(__name__)

def to_dense_batch(x: Tensor, batch: Optional[Tensor] = None,
                   fill_value: float = 0., max_num_nodes: Optional[int] = None,
                   batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    r"""Given a sparse batch of node features
    :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}` (with
    :math:`N_i` indicating the number of nodes in graph :math:`i`), creates a
    dense node feature tensor
    :math:`\mathbf{X} \in \mathbb{R}^{B \times N_{\max} \times F}` (with
    :math:`N_{\max} = \max_i^B N_i`).
    In addition, a mask of shape :math:`\mathbf{M} \in \{ 0, 1 \}^{B \times
    N_{\max}}` is returned, holding information about the existence of
    fake-nodes in the dense representation.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered. (default: :obj:`None`)
        fill_value (float, optional): The value for invalid entries in the
            resulting dense output tensor. (default: :obj:`0`)
        max_num_nodes (int, optional): The size of the output node dimension.
            (default: :obj:`None`)
        batch_size (int, optional) The batch size. (default: :obj:`None`)

    :rtype: (:class:`Tensor`, :class:`BoolTensor`)
    """
    if batch is None and max_num_nodes is None:
        mask = torch.ones(1, x.size(0), dtype=torch.bool, device=x.device)
        return x.unsqueeze(0), mask

    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_size is None:
        batch_size = int(batch.max()) + 1

    # torch_scatter.scatter_add: (src, index, dim=, out=, dim_size=)
    # "add source value to the given index"

    # torch.scatter_add: (input, dim, index, src)
    # new_ones: new array all-1 with size

    # default adds up to dim_size zero array

    num_nodes = x.new_zeros(x.size(0), dtype=torch.long)
    num_nodes = torch.scatter_add(num_nodes, -1, batch, batch.new_ones(x.size(0)) )
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    idx = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.size())[1:]
    out = x.new_full(size, fill_value)
    out[idx] = x
    out = out.view([batch_size, max_num_nodes] + list(x.size())[1:])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool,
                       device=x.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)

    return out, mask


def make_batch_vec(size_vec):
    batch_vector = []
    n=0
    for i in size_vec:
        for k in range(i):
            batch_vector.append(n)
        n+=1

    return torch.tensor(batch_vector)


def get_pair_dis_one_hot(d, bin_size=2, bin_min=-1, bin_max=30, num_classes=32):
    # without compute_mode='donot_use_mm_for_euclid_dist' could lead to wrong result.
    pair_dis = torch.cdist(d, d, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=num_classes)
    return pair_dis_one_hot


def masked_softmax(x, mask, **kwargs):
    x_masked = x.clone()
    x_masked[mask == 0] = -1.0e10 #-float("inf")

    return torch.softmax(x_masked, **kwargs)
