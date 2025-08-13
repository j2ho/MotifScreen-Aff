import torch

ScreeningLoss = torch.nn.BCEWithLogitsLoss()

# null gives ~ 0.1 when n == 5;  p=(0.5,0.0...) -> loss ~ 0.0275; p=(0.0,1.0,0.0...) -> loss ~ 0.25;
def RankingLoss( ps, qs ): #p: pred q:
    eps = 1.0e-6
    ps = torch.nn.functional.softmax(ps,dim=-1)
    qs = torch.nn.functional.softmax(qs,dim=-1)
    ps = torch.clamp(ps, min=eps, max=1.0-eps)
    qs = torch.clamp(qs, min=eps, max=1.0-eps)
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


def MaskedBCE(cats,preds,masks,debug=False):
    device = masks.device

    lossP = torch.tensor(0.0).to(device)
    lossN = torch.tensor(0.0).to(device)

    for cat,mask,pred in zip(cats,masks,preds):
        # "T": NTYPES; "N": ngrids
        # cat: NxT
        # mask : N
        # pred : NxT
        ngrid = cat.shape[0]

        Q = pred[-ngrid:]

        a = -cat*torch.log(Q+1.0e-6) #-PlogQ
        icat = (cat<0.001).float()

        #catG = grouped_cat(cat,device) # no such thing in ligand
        #g = -catG*torch.log(Q+1.0e-6) # on group-cat
        #icatG = (catG<0.001).float()

        # transformed iQ -- 0~0.5->1, drops to 0 as x = 0.5->1.0
        # allows less penalty if x is 0.0 ~ 0.5
        #iQt = -0.5*torch.tanh(5*Q-3.0)+1)
        iQt = torch.clamp(1.0-Q, min=1.0e-5)
        b  = -icat*torch.log(iQt) #penalize if high

        # normalize by num grid points & cat points
        norm = 1.0

        lossP += torch.sum(torch.matmul(mask, a))*norm
        lossN += torch.sum(torch.matmul(mask, b))*norm

    return lossP, lossN


def ContrastLoss(preds,masks):
    loss = torch.tensor(0.0).to(masks.device)

    for mask,pred in zip(masks,preds):
        imask = 1.0 - mask
        ngrid = mask.shape[0]
        psum = torch.sum(torch.matmul(imask,pred[-ngrid:]))/ngrid

        loss += psum
    return loss


def StructureLoss( Yrec, Ylig, nK, opt='mse'):
    # Yrec: [B,Kmax,3] Ylig: [1,K,3]
    k = nK[0].item()
    dY = Yrec[0, :k, :] - Ylig[0, :k, :]  # More explicit indexing
    N = 1
    if opt == 'mse':
        loss1 = torch.sum(dY*dY,dim=0) # sum over K
        loss1_sum = torch.sum(loss1)/N
    elif opt == 'Huber':
        d = torch.sqrt(torch.sum(dY*dY,dim=-1))/nK[0] #distance-per-K
        loss1_sum = 10.0*torch.nn.functional.huber_loss(d,torch.zeros_like(d))

    mae = torch.sum(torch.abs(dY))/nK[0] # this is correct mae...
    return loss1_sum, mae

def PairDistanceLoss( Dpred, X, nK, bin_min = -0.1, bin_size=0.25, bin_max=15.75 ):
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
def SpreadLoss(Ylig, A, grid, nK, sig=2.0): #Ylig:label(B x K x 3), A:attention (B x Nmax x K), grid: B x Nmax x 3
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
def SpreadLoss_v2(Ylig, A, grid, nK, sig=2.0):
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
