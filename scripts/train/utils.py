import torch
import numpy as np

def calc_AUC(Pt, Pf):
    Pt = np.array(Pt)
    Pf = np.array(Pf)

    if Pt.size == 0 or Pf.size == 0:
        return -1.0

    comparison = Pt[:, None] - Pf[None, :]
    count = np.sum(comparison > 0) + 0.5 * np.sum(comparison == 0)

    auc = count / (Pt.size * Pf.size)
    return auc

def count_parameters(model):
    #print([p.numel() for p in model.parameters()])
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_cuda(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, tuple):
        return (to_cuda(v, device) for v in x)
    elif isinstance(x, list):
        return [to_cuda(v, device) for v in x]
    elif isinstance(x, dict):
        return {k: to_cuda(v, device) for k, v in x.items()}
    else:
        # DGLGraph or other objects
        return x.to(device=device)
