import torch

ScreeningLoss = torch.nn.BCEWithLogitsLoss()

def ScreeningLossW( aff, label, weight=None ):
    lossfunc = torch.nn.BCEWithLogitsLoss( reduction='none' )
    loss = lossfunc( aff, label )
    if weight != None:
        loss = weight*loss

    #print( aff, label, lossfunc( aff, label ), loss)

    return loss.mean()

# null gives ~ 0.1 when n == 5;  p=(0.5,0.0...) -> loss ~ 0.0275; p=(0.0,1.0,0.0...) -> loss ~ 0.25;
def RankingLoss( ps, qs ): #p: pred q:
    eps = 1.0e-6
    ps = torch.nn.functional.softmax(ps,dim=-1)+eps
    qs = torch.nn.functional.softmax(qs,dim=-1)+eps
    loss = torch.sum(ps*torch.log(ps/qs + eps))
    return loss

def ScreeningContrastLoss( embs, blabel, nK ):
    # embs: B x k
    # blabel: B
    # what should be ideal value
    loss = torch.tensor(0.0).to(embs.device)
    for emb,l,n in zip(embs,blabel,nK): #different number of Ks
        diff = emb[:n] - l
        loss = loss + torch.dot(diff,diff) # 0 or 1

    return loss


def AffinityLoss(pAff_pred, pAff, blabel=None):
    """
    Affinity loss function for values in the range of 1-15

    Args:
        pAff_pred: Predicted affinity values [batch_size]
        pAff: True affinity values, with placeholders (-1) for decoys [batch_size]
        blabel: Binary labels (1 for active, 0 for decoys) [batch_size]

    Returns:
        loss: The computed loss
    """
    if pAff_pred.dim() > 1:
        pAff_pred = pAff_pred.squeeze(-1)

    # Create mask for active compounds
    if blabel is not None:
        active_mask = (blabel == 1).float()
    else:
        active_mask = (pAff > 0).float()

    # Replace placeholder values with zeros for loss calculation
    valid_pAff = pAff * active_mask

    # Use Huber loss for robustness to outliers
    huber = torch.nn.HuberLoss(reduction='none')
    affinity_loss_per_sample = huber(pAff_pred, valid_pAff) * active_mask

    # Normalize by number of active compounds
    num_active = torch.sum(active_mask) + 1e-8
    affinity_loss = torch.sum(affinity_loss_per_sample) / num_active

    return affinity_loss


def CustomAffinityLoss(pAff_pred, pAff, blabel, affinity_weight=1.0, margin=15.0):
    """
    Custom loss function for binding affinity prediction with single active per batch

    Args:
        pAff_pred: Predicted affinity values [batch_size]
        pAff: True affinity values, with placeholders (-1) for decoys [batch_size]
        blabel: Binary labels (1 for active, 0 for decoys) [batch_size]
        affinity_weight: Weight for affinity component vs. ranking component
        margin: Margin for ranking loss between active and decoys

    Returns:
        total_loss: The computed loss value
    """
    # 1. Affinity prediction loss (only for active compound)
    if torch.isnan(pAff_pred).any(): print("NaN in pAff_pred")
    if torch.isinf(pAff_pred).any(): print("Inf in pAff_pred")
    if torch.isnan(pAff).any(): print("NaN in pAff")
    if torch.isinf(pAff).any(): print("Inf in pAff")

    # Mask for active compound
    valid_mask = (pAff >=0.0) & ~torch.isnan(pAff)
    active_mask = (blabel == 1)
    decoy_mask = (blabel == 0).float() 
    pred = pAff_pred
    # print ("pred", pAff_pred)
    target = torch.where(valid_mask, pAff, torch.zeros_like(pAff))
    # print("target", target)
    # Apply Huber loss only to active compound
    huber = torch.nn.HuberLoss(reduction='none')
    # Only compare with actual affinity values (not the -1 placeholders)
    affinity_loss_per_sample = huber(pred, target) 

    mask = valid_mask & active_mask
    affinity_loss_per_sample = affinity_loss_per_sample[mask]
    # print("affinity_loss_per_sample", affinity_loss_per_sample)
    affinity_loss = torch.sum(affinity_loss_per_sample) / (mask.sum() + 1e-8)
    # print("affinity_loss", affinity_loss)
    if affinity_weight == 1.0:
        return affinity_loss
    # 2. Ranking loss (active compound should have higher predicted affinity than decoys)
    # Get the prediction for the active compound
    active_pred = torch.sum(transformed_pred * active_mask)
    # print("active_pred", active_pred)

    # Compare active prediction with each decoy prediction using a margin
    # For each decoy, we want: active_pred > decoy_pred + margin
    decoy_preds = transformed_pred * decoy_mask
    # print("decoy_preds", decoy_preds)
    if len(blabel) > 4 and active_mask[0] == 1: # has active and at least three decoys
        ranking_losses = torch.relu(decoy_preds - active_pred + margin) * decoy_mask
    # print("ranking_losses", ranking_losses)
        ranking_loss = torch.sum(ranking_losses) / (torch.sum(decoy_mask) + 1e-8)
    else: 
        ranking_loss = None
    # print("ranking_loss", ranking_loss)
    # Combine losses
    if ranking_loss:
        total_loss = affinity_weight * affinity_loss + (1.0 - affinity_weight) * ranking_loss
    else:
        total_loss = affinity_loss
    return total_loss


### loss calculation functions
def grouped_cat(cat):
    import src.src_Grid.motif as motif
    import numpy as np

    catG = torch.zeros(cat.shape).to(device)

    # iter over 6 groups
    for k in range(1,7):
        js = np.where(np.array(motif.SIMPLEMOTIFIDX)==k)[0]
        if len(js) > 0:
            a = torch.max(cat[:,js],dim=1)[0]
            if max(a).float() > 0.0:
                for j in js: catG[:,j] = a

    # normalize
    norm = torch.sum(catG, dim=1)[:,None].repeat(1,NTYPES)+1.0e-6
    catG = catG / norm

    return catG

def MaskedBCE(cats,preds,masks,debug=False):
    device = masks.device

    lossC = torch.tensor(0.0).to(device)
    lossG = torch.tensor(0.0).to(device)
    lossR = torch.tensor(0.0).to(device)

    # iter through batches (actually not)
    bygrid = [0.0, 0.0, 0.0]
    for cat,mask,pred in zip(cats,masks,preds):
        # "T": NTYPES; "N": ngrids
        # cat: NxT
        # mask : N
        # pred : NxT
        ngrid = cat.shape[0]

        Q = pred[-ngrid:]

        a = -cat*torch.log(Q+1.0e-6) #-PlogQ
        # old -- cated ones still has < 1.0 thus penalized
        #b = -(1.0-cat)*torch.log((1.0-Q)+1.0e-6)
        icat = (cat<0.001).float()

        #catG = grouped_cat(cat,device) # no such thing in ligand
        #g = -catG*torch.log(Q+1.0e-6) # on group-cat
        #icatG = (catG<0.001).float()

        # transformed iQ -- 0~0.5->1, drops to 0 as x = 0.5->1.0
        # allows less penalty if x is 0.0 ~ 0.5
        #iQt = -0.5*torch.tanh(5*Q-3.0)+1)
        iQt = 1.0-Q+1.0e-6
        b  = -icat*torch.log(iQt) #penalize if high

        # normalize by num grid points & cat points
        norm = 1.0

        lossC += torch.sum(torch.matmul(mask, a))*norm
        #lossG += torch.sum(torch.matmul(mask, g))*norm
        lossR += torch.sum(torch.matmul(mask, b))*norm


        bygrid[0] += torch.mean(torch.matmul(mask, a)).float()
        #bygrid[1] += torch.mean(torch.matmul(mask, g)).float()
        bygrid[2] += torch.mean(torch.matmul(mask, b)).float()

        if debug:
            print("Label/Mask/Ngrid/Norm: %.1f/%d/%d/%.1f"%(float(torch.sum(cat)), int(torch.sum(mask)), ngrid, float(norm)))
    return lossC, lossG, lossR, bygrid

def ContrastLoss(preds,masks):
    loss = torch.tensor(0.0).to(masks.device)

    for mask,pred in zip(masks,preds):
        imask = 1.0 - mask
        ngrid = mask.shape[0]
        psum = torch.sum(torch.matmul(imask,pred[-ngrid:]))/ngrid

        loss += psum
    return loss

def structural_loss( Yrec, Ylig, nK, opt='mse' ):
    # Yrec: BxKx3 Ylig: K x 3

    dY = Yrec[0,:nK[0],:] - Ylig[0] # hack

    N = 1
    if opt == 'mse':
        loss1 = torch.sum(dY*dY,dim=0) # sum over K
        loss1_sum = torch.sum(loss1)/N
    elif opt == 'Huber':
        d = torch.sqrt(torch.sum(dY*dY,dim=-1))/nK[0] #distance-per-K
        loss1_sum = 10.0*torch.nn.functional.huber_loss(d,torch.zeros_like(d))

    mae = torch.sum(torch.abs(dY))/nK[0] # this is correct mae...

    return loss1_sum, mae

def distance_loss( Dpred, X, nK, bin_min = -1, bin_size=0.5, bin_max=30 ):
    # make label first
    #X: label coordinate
    pair_dis = torch.cdist(X, X, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    pair_dis_one_hot = torch.nn.functional.one_hot(pair_dis_bin_index, num_classes=Dpred.shape[-1]).float()

    LossFunc = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = torch.tensor(0.0).to(X.device)
    for pred,label,k in zip(Dpred,pair_dis_one_hot,nK):
        # move channel dimension (2nd) to 1st dim
        pred = pred[:k,:k,:].transpose(1,2)
        label = label[:k,:k,:].transpose(1,2)
        loss1 = LossFunc(pred, label)
        loss = loss + loss1
        #print(loss1, torch.argmax(pred[:k,:k,:],dim=-1), torch.argmax(label[:k,:k,:],dim=-1))
    return loss

def distance_loss2( Dpred, X, nK, bin_min = -0.1, bin_size=0.25, bin_max=15.75 ):
    # make label first
    #X: label coordinate
    pair_dis = torch.cdist(X, X, compute_mode='donot_use_mm_for_euclid_dist')
    pair_dis[pair_dis>bin_max] = bin_max
    pair_dis_bin_index = torch.div(pair_dis - bin_min, bin_size, rounding_mode='floor').long()
    LossFunc1 = torch.nn.CrossEntropyLoss() #reduction='sum')
    LossFunc2 = torch.nn.HuberLoss()

    loss = torch.tensor(0.0).to(X.device)
    for pred,label,d_p,d_l,k in zip(Dpred,pair_dis_bin_index, Dpred, pair_dis,nK):
        # move channel dimension (2nd) to 1st dim
        dbins = torch.arange(0.0,15.9,0.25).to( X.device )
        d_p = torch.einsum('k,ijk->ij',dbins,pred[:k,:k])
        pred = pred[:k,:k,:].transpose(1,2)
        label = label[:k,:k]
        loss1 = LossFunc1(pred, label)
        loss2 = LossFunc2(d_p[:k,:k], d_l[:k,:k])
        loss = loss + loss1 + loss2
        #print(loss1, torch.argmax(pred[:k,:k,:],dim=-1), torch.argmax(label[:k,:k,:],dim=-1))
    return loss

###
def spread_loss(Ylig, A, grid, nK, sig=2.0): #Ylig:label(B x K x 3), A:attention (B x Nmax x K), grid: B x Nmax x 3
    # actually B == 1
    loss2 = torch.tensor(0.0)

    for b, (x,k) in enumerate( zip(grid, nK) ): #ngrid: B x maxn
        n = x.shape[0]
        #z = A[0,:n,:Ylig.shape[0]] # N x K
        z = A[0,:n,:k] # N x K
        x = x[:,None,:]

        dX = x-Ylig
        overlap = torch.exp(-torch.sum(dX*dX,axis=-1)/(sig*sig)) # N x K
        if z.shape[0] != overlap.shape[0]: continue

        loss2 = loss2 - torch.sum(overlap*z)

    #loss2 = -torch.sum(overlap*z)

    return loss2 # max -(batch_size x K)

# penalty
#Ylig:label(B x K x 3), A:attention (B x Nmax x K), grid: B x Nmax x 3
def spread_loss2(Ylig, A, grid, nK, sig=2.0):
    # actually B == 1
    loss2 = torch.tensor(0.0)

    for b, (x,k) in enumerate( zip(grid, nK) ): #ngrid: B x maxn
        n = x.shape[0]
        z = A[0,:n,:k] # N x K
        x = x[:,None,:] # N x 1 x 3

        dX = (x-Ylig)/sig # N x K x 3
        dev = torch.sum(dX*dX, axis=-1) # N x K

        if z.shape[0] != dev.shape[0]: continue

        loss2 = loss2 + torch.sum(dev*z)

    return loss2 # max -(batch_size x K)

'''
def spread_loss( Ylig, A, grid, nK, sig=2.0): #key(B x K x 3), attention (B x Nmax x K)
    loss2 = torch.tensor(0.0)
    i = 0
    N = A.shape[0]

    for b, (x,k,y) in enumerate( zip(grid, nK, Ylig) ): #ngrid: B x maxn
        n = x.shape[0]
        z = A[b,:n,:k] # Nmax x Kmax

        dX = x[:,None,:] - y[None,:k,:]

        overlap = torch.exp(-torch.sum(dX*dX,axis=-1)/(sig*sig)) # N x K
        if z.shape[0] != overlap.shape[0]: continue

        loss2 = loss2 - torch.sum(overlap*z)

        i += n
    return loss2 # max -(batch_size x K)
'''
