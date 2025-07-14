#!/usr/bin/env python
import os, sys
import numpy as np
from os.path import join, isdir
import torch
import time
import dgl

## DDP related modules
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataset import collate, DataSet
from src.model.models.msk1 import EndtoEndModel as MSK_1
from src.model.models.msk2 import MSK_ScreenAff as MSK_2

from scripts.train.utils import count_parameters, to_cuda, calc_AUC
import src.model.loss.losses as Loss
from scripts.train.configs import args_graphfix34 as args #same hyperparm

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")

# default setup
set_params={
    'datapath' : "/ml/motifnet/features_com2/",
    'ball_radius'  : 8.0,
    'edgedist'     : (2.2,4.5), # unused! grid: 18 neighs -- exclude cube-edges
    'edgemode'     : 'topk',
    'edgek'        : (8,16),
    "randomize"    : 0.2, # Ang, pert the rest
    #"randomize_lig": 0.5, # Ang, pert the motif coord! #reduce noise...
    "ntype"        : 6,
    'debug'        : ('-debug' in sys.argv),
    'maxedge'      : 40000,
    'maxnode'      : 3000,
    'drop_H'       : True,
    'max_subset'   : 5,
    'load_cross'   : args.load_cross,
    'cross_grid'   : args.cross_grid,
    'cross_eval_struct' : args.cross_eval_struct,
    'nonnative_struct_weight' : args.nonnative_struct_weight,
    'input_features': args.input_features,
    'firstshell_as_grid': args.firstshell_as_grid
}

params_loader={
    'shuffle': (not args.ddp), 
    'num_workers':5 if not args.debug else 1,
    'pin_memory':True,
    'collate_fn':collate,
    'batch_size':1 if not args.debug else 1}

if not args.ddp:
    rank = 0

### load_params / making model,optimizer,loss,etc.
def load_params(rank):
    device = torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    ## model
    model = MSK_1(args) 
    model.to(device)

    ## loss
    train_loss_empty={"total":[],"BCEc":[],"BCEg":[],"BCEr":[],"contrast":[],"reg":[], "struct":[], "mae":[], "spread":[], "Screen":[], "ScreenC":[], "ScreenR":[], "Dkey":[]}
    valid_loss_empty={"total":[],"BCEc":[],"BCEg":[],"BCEr":[],"contrast":[],"struct":[], "mae":[], "spread":[], "Screen":[], "ScreenC":[], "ScreenR":[], "Dkey":[]}
    
    epoch=0
    ## optimizer
    optimizer=torch.optim.Adam(model.parameters(),lr=args.LR)

    if os.path.exists("models/%s/model.pkl"%args.modelname):
        if not args.silent: print("Loading a checkpoint")
        checkpoint = torch.load(join("models", args.modelname, "model.pkl"),map_location=device)

        trained_dict = {}
        model_dict = model.state_dict()
        model_keys = list(model_dict.keys())
        
        for key in checkpoint["model_state_dict"]:
            if key.startswith("module."):
                newkey = key.replace('module.','se3_Grid.')
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
            elif "tranistion." in key:
                newkey = key.replace('tranistion.','transition.')
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
            else:
                if key in model_keys:
                    wts = checkpoint["model_state_dict"][key]
                    if wts.shape == model_dict[key].shape: # load only if has the same shape
                        trained_dict[key] = wts
                    else:
                        print("skip", key)

        nnew, nexist = 0,0
        for key in model_keys:
            if key not in trained_dict:
                nnew += 1
                print("new", key)
            else:
                nexist += 1
        if rank == 0: print("params", nnew, nexist)
        
        model.load_state_dict(trained_dict, strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint["epoch"]+1 
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        for key in train_loss_empty:
            if key not in train_loss: train_loss[key] = []
        for key in valid_loss_empty:
            if key not in valid_loss: valid_loss[key] = []
            
        if not args.silent: print("Restarting at epoch", epoch)
        
    else:
        if not args.silent: print("Training a new model")
        train_loss = train_loss_empty
        valid_loss = valid_loss_empty
    
        best_models = []
        if not isdir(join("models", args.modelname)):
            if not args.silent: print("Creating a new dir at", join("models", args.modelname))
            os.mkdir(join("models", args.modelname))

    # temporary
    # re-initialize class module
    if epoch == 0:
        for i, (name, layer) in enumerate(model.named_modules()):
            if hasattr(layer,'weight'):
                if layer.weight != None:
                    print(name, layer.weight.data.numel())

            if isinstance(layer,torch.nn.Linear) and \
               ("class" in name or 'Xform' in name): #".wr" in name or '.wl' in name:
                if rank == 0:
                    print("reweight", name)
                layer.weight.data *= 0.1

    if rank == 0:
        print("Nparams:",count_parameters(model))
        print("Loaded")

    return model,optimizer,epoch,train_loss,valid_loss

def load_data(txt, world_size, rank):
    from torch.utils import data

    target_s = []
    ligands_s = []
    is_ligand_s = []
    decoy_npzs = []
    weights = {}
    print(txt)
    for ln in open(txt,'r'):
        if ln.startswith('#'): continue
        x = ln[:-1].split()
        is_ligand = bool(x[0]) #0:PP; 1:PL
        target    = x[1]
        mol2type  = x[2] #how to read mol2f: single/batch
        mol2f     = x[3] #.npz or .mol2
        activemol = x[4] # active molecule name or selection logic
        decoyf    = x[5] #.npz or batch-mol2
        if len(x) > 6:
            weights[target.split('/')[-1]] = float(x[6])
        else:
            weights[target.split('/')[-1]] = 1.0
        
        is_ligand_s.append(is_ligand)
        target_s.append(target)
        ligands_s.append((mol2type,mol2f,activemol,decoyf))
        if decoyf.endswith('.npz') and decoyf not in decoy_npzs:
            decoy_npzs.append(decoyf)
        
    data_set = DataSet(target_s, is_ligand_s, ligands_s, decoy_npzs=decoy_npzs, **set_params)

    if args.ddp:
        sampler = data.distributed.DistributedSampler(data_set,num_replicas=world_size,rank=rank)
        data_loader = data.DataLoader(data_set,sampler=sampler,**params_loader)
    else:
        data_loader = data.DataLoader(data_set, **params_loader)
    return data_loader, weights

### train_model
def train_model(rank,world_size,dumm):
    gpu=rank%world_size
    dist.init_process_group(backend='gloo',world_size=world_size,rank=rank)

    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")
    torch.cuda.set_device(device)

    ## load_params
    model,optimizer,start_epoch,train_loss,valid_loss=load_params(rank)

    if args.ddp:
        ddp_model=DDP(model,device_ids=[gpu],find_unused_parameters=False)

    ## data loader
    train_loader, weights_train = load_data(args.datasetf[0], world_size, rank)
    valid_loader, weights_valid = load_data(args.datasetf[1], world_size, rank)

    auc_train = {'CHEMBL':[],'DUDE':[],'PDB':[]}
    auc_valid = {'CHEMBL':[],'DUDE':[],'PDB':[]}
    
    ## iteration
    for epoch in range(start_epoch,args.max_epoch):
        ## train
        if args.ddp:
            ddp_model.train()
            temp_loss, Pt, Pf =train_one_epoch(ddp_model,optimizer,train_loader,rank,epoch,True,w=weights_train)
        else:
            model.train()
            temp_loss, Pt, Pf =train_one_epoch(model,optimizer,train_loader,rank,epoch,True,w=weights_train)
            
        for k in train_loss:
            train_loss[k].append(np.array(temp_loss[k]))
        
        if rank == 0:
            for key in Pt:
                if len(Pt[key]) > 10 and len(Pf[key]) > 10:
                    auc_train[key].append(calc_AUC(Pt[key],Pf[key]))
            
        optimizer.zero_grad()
        ## evaluate
        with torch.no_grad():
            if args.ddp:
                ddp_model.eval()
                temp_loss, Pt, Pf = train_one_epoch(ddp_model,optimizer,valid_loader,rank,epoch,False,w=weights_valid)
            else:
                temp_loss, Pt, Pf = train_one_epoch(model,optimizer,valid_loader,rank,epoch,False,w=weights_valid)
        
            for k in valid_loss:
                valid_loss[k].append(np.array(temp_loss[k]))

            if rank == 0:
                for key in Pt:
                    if len(Pt[key]) > 10 and len(Pf[key]) > 10:
                        auc_valid[key].append(calc_AUC(Pt[key],Pf[key]))

        print("***SUM***")
        print("Train loss | %7.4f | Valid loss | %7.4f"%((np.mean(train_loss['total'][-1]),np.mean(valid_loss['total'][-1]))))

        if rank == 0:
            auc_l = "AUC: "
            for key in ['PDB','CHEMBL','DUDE']:
                if key in auc_train and len(auc_train[key]) > 0:
                    auc_l += f' {key} {auc_train[key][-1]:6.4f}'
                if key in auc_valid and len(auc_valid[key]) > 0:
                    auc_l += f' / {auc_valid[key][-1]:6.4f}'
            print(auc_l)

        ## update the best model
        if rank==0:
            if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'auc_train': auc_train,
                    'auc_valid': auc_valid,
                }, join("models", args.modelname, "best.pkl"))
   
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'auc_train': auc_train,
                'auc_valid': auc_valid,
                
            }, join("models", args.modelname, "model.pkl"))

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'auc_train': auc_train,
                'auc_valid': auc_valid,
                
            }, join("models", args.modelname, "epoch%d.pkl"%epoch))
            

### train_one_epoch
def train_one_epoch(model,optimizer,loader,rank,epoch,is_train,w):
    temp_loss={"total":[], "BCEc":[], "BCEg":[], "BCEr":[], "contrast":[], "reg":[], "struct":[], "mae":[], "spread":[], "Screen":[], "ScreenC":[], "ScreenR":[], "Dkey":[]}
    b_count,e_count=0,0
    accum=1
    device=torch.device("cuda:%d"%rank if (torch.cuda.is_available()) else "cpu")

    Pt = {'CHEMBL':[],'DUDE':[],'PDB':[]}
    Pf = {'CHEMBL':[],'DUDE':[],'PDB':[]}
    
    for i, inputs in enumerate(loader):
        if inputs == None:
            e_count += 1
            continue

        (Grec, Glig, cats, masks, keyxyz, keyidx, blabel, info) = inputs
        if Grec == None:
            e_count += 1
            continue

        with torch.cuda.amp.autocast(enabled=False):
            with model.no_sync(): #should be commented if
            #if True:
                t0 = time.time()
                
                if Glig != None:
                    Glig = to_cuda(Glig,device)
                    keyxyz = to_cuda(keyxyz, device)
                    keyidx = to_cuda(keyidx, device)
                    nK = info['nK'].to(device)
                    blabel = to_cuda(blabel, device)
                else:
                    keyxyz, keyidx, nK, blabel = None, None, None, None                    
                    
                Grec = to_cuda(Grec, device)
                pnames  = info["pname"]
                grid = info['grid'].to(device)
                eval_struct = info['eval_struct'][0]
                grididx = info['grididx'].to(device)

                # Ggrid memory check -- otherwise 'x' and 'nsize' is sufficient
                t1 = time.time()
                Yrec_s, Dkey_s, z, MotifP, aff = model(Grec, 
                                                       Glig, keyidx, grididx,
                                                       gradient_checkpoint=is_train,
                                                       drop_out=is_train)
                
                if MotifP == None:
                    continue
        
                ## 1. GridNet loss related
                lossGc, lossGg, lossGr = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
                lossGcontrast = torch.tensor(0.0).to(device)
                p_reg = torch.tensor(0.).to(device)
                
                if cats != None:
                    cats = to_cuda(cats, device)
                    masks = to_cuda(masks, device)
                    
                    MotifP = torch.sigmoid(MotifP) # Then convert to sigmoid (0~1)
                    MotifPs = [MotifP] # assume batchsize=1
                    t1 = time.time()
                
                    # 1-1. GridNet main losses; c-category g-group r-reverse contrast-contrast
                    
                    lossGc,lossGg,lossGr,bygrid = Loss.MaskedBCE(cats,MotifPs,masks)
                    lossGr = args.w_false*lossGr
                    lossGcontrast = args.w_contrast*Loss.ContrastLoss(MotifPs,masks) # make overal prediction low as possible

                    p_reg = torch.nn.functional.relu(torch.sum(MotifP*MotifP-25.0))
                
                ## 2. TRnet loss starts here
                Pbind = [] #verbose
                lossTs, mae, lossTr, lossTd = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)
                lossScreen, lossScreenC, lossScreenR = torch.tensor(0.0).to(device), torch.tensor(0.0).to(device), torch.tensor(0.0).to(device)

                if Yrec_s != None and grid.shape[1] == z.shape[1]:
                    try:
                    #if True:
                        # 2-1. structural loss
                        if len(nK.shape) > 1:
                            nK = nK.squeeze(dim=0)

                        if eval_struct:
                            lossTs, mae = Loss.structural_loss( Yrec_s, keyxyz, nK, opt=args.struct_loss ) #both are Kx3 coordinates
                            lossTd = args.w_Dkey*Loss.distance_loss2( Dkey_s, keyxyz, nK )
                            
                            lossTr = args.w_spread*Loss.spread_loss( keyxyz, z, grid, nK )
                            lossTr2 = args.w_spread*Loss.spread_loss2( keyxyz, z, grid, nK )
                            lossTr = lossTr + 0.2*lossTr2
                            
                        # 2-2.s screening loss
                        lossScreen = Loss.ScreeningLoss( aff[0], blabel )
                        lossScreenR = Loss.RankingLoss( torch.sigmoid(aff[0]), blabel )
                        lossScreenC = Loss.ScreeningContrastLoss( aff[1], blabel, nK )
                        Pbind = ['%4.2f'%float(a) for a in torch.sigmoid(aff[0])]

                        key = 'PDB'
                        if info['ligands'][0][0].startswith('CHEMBL') and pnames[0][0] in ['O','P','Q']: key = 'CHEMBL'
                        elif info['ligands'][0][0].startswith('CHEMBL'): key = 'DUDE'
                        
                        Pt[key].append(float(torch.sigmoid(aff[0][0]).cpu()))
                        Pf[key] += list(torch.sigmoid(aff[0][1:]).cpu().detach().numpy())
                    except:
                    #else:
                        pass


                t2 = time.time()
                    
                ## 3. Full regularizer
                l2_reg = torch.tensor(0.).to(device)
                if is_train:
                    for param in model.parameters(): l2_reg += torch.norm(param)
                
                ## final loss
                ## default loss
                loss = args.wGrid*(lossGc + lossGg + lossGr + lossGcontrast + \
                                   args.w_reg*(l2_reg+p_reg))  \
                      + args.wTR*(lossTs + lossTr + lossTd) \
                      + args.w_screen*lossScreen + args.w_screen_ranking*lossScreenR + args.w_screen_contrast*lossScreenC

                trg_weight = w[pnames[0]] if pnames[0] in w else 1.0
                loss = loss*trg_weight
                
                #store as per-sample loss
                temp_loss["total"].append(loss.cpu().detach().numpy()) 
                temp_loss["BCEc"].append(lossGc.cpu().detach().numpy()) 
                temp_loss["BCEg"].append(lossGg.cpu().detach().numpy()) 
                temp_loss["BCEr"].append(lossGr.cpu().detach().numpy()) 
                temp_loss["contrast"].append(lossGcontrast.cpu().detach().numpy())
                temp_loss["reg"].append((p_reg+l2_reg).cpu().detach().numpy())
                if lossTs > 0.0:
                    temp_loss["struct"].append(lossTs.cpu().detach().numpy())
                    temp_loss["mae"].append(mae.cpu().detach().numpy())
                    temp_loss["spread"].append(lossTr.cpu().detach().numpy())
                    temp_loss["Dkey"].append(lossTd.cpu().detach().numpy())
                if lossScreen > 0.0:
                    temp_loss["Screen"].append(lossScreen.cpu().detach().numpy())
                    temp_loss["ScreenR"].append(lossScreenR.cpu().detach().numpy())
                    temp_loss["ScreenC"].append(lossScreenC.cpu().detach().numpy())
            
            # Only update after certain number of accululations.
            gridinfo = info['gridinfo'][0].split('/')[-1].replace('.grid.npz','')
            if is_train and (b_count+1)%accum == 0:
                loss.requires_grad_(True)
                loss.backward()
                optimizer.step()    
                optimizer.zero_grad()
                print("Rank %d TRAIN Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %8.3f (M/R/C/S(mae)/Sp/Scr: %6.1f/%6.1f/%6.3f|%6.1f(%5.2f)/%6.1f/%6.1f|%4.2f/%4.2f), %s:%s"
                      %(rank,args.modelname, epoch, args.max_epoch, b_count, len(loader),
                        np.sum(temp_loss["total"][-1*accum:]),
                        np.sum(temp_loss["BCEc"][-1*accum:]),
                        np.sum(temp_loss["BCEr"][-1*accum:]),
                        np.sum(temp_loss["contrast"][-1*accum:]),
                        float(lossTs),
                        float(mae),
                        float(lossTr),
                        float(lossTd),
                        np.sum(temp_loss["Screen"][-1*accum:]),
                        np.sum(temp_loss["ScreenR"][-1*accum:]),
                        pnames[0], gridinfo
                      ), ' '.join(Pbind), ','.join(info['ligands'][0]))
                
            elif (b_count+1)%accum == 0: # valid
                print("Rank %d VALID Epoch(%s): [%2d/%2d], Batch: [%2d/%2d], loss: %8.3f (M/R/C|S(mae)/Sp/D|Scr: %6.1f/%6.1f/%6.3f|%6.1f(%5.2f)/%6.1f/%6.1f|%4.2f/%4.2f), %s:%s"
                      %(rank,args.modelname, epoch, args.max_epoch, b_count, len(loader),
                        np.sum(temp_loss["total"][-1*accum:]),
                        np.sum(temp_loss["BCEc"][-1*accum:]),
                        np.sum(temp_loss["BCEr"][-1*accum:]),
                        np.sum(temp_loss["contrast"][-1*accum:]),
                        float(lossTs),
                        float(mae),
                        float(lossTr),
                        float(lossTd),
                        np.sum(temp_loss["Screen"][-1*accum:]),
                        np.sum(temp_loss["ScreenR"][-1*accum:]),
                        pnames[0], gridinfo
                      ), ' '.join(Pbind), ','.join(info['ligands'][0]))
                t3 = time.time()
                        
            b_count += 1

    return temp_loss, Pt, Pf

## main
if __name__=="__main__":
    print("dgl version", dgl.__version__)
    torch.cuda.empty_cache()
    mp.freeze_support()
    world_size=torch.cuda.device_count()
    print("Using %d GPUs.."%world_size)
    
    if ('MASTER_ADDR' not in os.environ):
        os.environ['MASTER_ADDR'] = 'localhost' # multinode requires this set in submit script
    if ('MASTER_PORT' not in os.environ):
        os.environ['MASTER_PORT'] = '12346'

    os.system("touch GPU %d"%world_size)

    if args.ddp:
        mp.spawn(train_model,args=(world_size,0),nprocs=world_size,join=True)
    else:
        train_model(0, 1, None)