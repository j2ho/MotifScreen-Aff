#!/usr/bin/env python
"""
Updated training script for MotifScreen-Aff using new YAML-based configuration system
"""
import os
import sys
import numpy as np
from os.path import join, isdir
import torch
import time
import dgl
import argparse

# DDP related modules
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.dataset import collate, DataSet
from src.model.models.msk1 import EndtoEndModel as MSK_1
from src.model.models.msk2 import MSK_ScreenAff as MSK_2

from scripts.train.utils import count_parameters, to_cuda, calc_AUC
import src.model.loss.losses as Loss
from configs.config import create_common_config, load_config_with_base

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")


def create_dataset_params(config):
    """Create dataset parameters from config"""
    return {
        'datapath': config.data.datapath,
        'ball_radius': config.data.ball_radius,
        'edgedist': tuple(config.data.edgedist),
        'edgemode': config.data.edgemode,
        'edgek': tuple(config.data.edgek),
        'randomize': config.data.randomize,
        'ntype': config.data.ntype,
        'debug': config.debug,
        'maxedge': config.data.maxedge,
        'maxnode': config.data.maxnode,
        'drop_H': config.data.drop_H,
        'max_subset': config.data.max_subset,
        'load_cross': config.load_cross,
        'cross_grid': config.cross_grid,
        'cross_eval_struct': config.cross_eval_struct,
        'nonnative_struct_weight': config.nonnative_struct_weight,
        'input_features': config.input_features,
        'firstshell_as_grid': config.firstshell_as_grid
    }


def create_dataloader_params(config):
    """Create dataloader parameters from config"""
    return {
        'shuffle': config.dataloader.shuffle,
        'num_workers': config.dataloader.num_workers,
        'pin_memory': config.dataloader.pin_memory,
        'collate_fn': collate,
        'batch_size': config.dataloader.batch_size
    }


def load_params(rank, config):
    """Load model, optimizer, and training state"""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = MSK_1(config)
    model.to(device)
    
    # Initialize loss tracking
    train_loss_empty = {
        "total": [], "BCEc": [], "BCEg": [], "BCEr": [], "contrast": [], 
        "reg": [], "struct": [], "mae": [], "spread": [], "Screen": [], 
        "ScreenC": [], "ScreenR": [], "Dkey": []
    }
    valid_loss_empty = {
        "total": [], "BCEc": [], "BCEg": [], "BCEr": [], "contrast": [], 
        "struct": [], "mae": [], "spread": [], "Screen": [], 
        "ScreenC": [], "ScreenR": [], "Dkey": []
    }
    
    epoch = 0
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    
    # Load checkpoint if it exists
    checkpoint_path = join("models", config.modelname, "model.pkl")
    if os.path.exists(checkpoint_path):
        if not config.silent:
            print("Loading a checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        trained_dict = {}
        model_dict = model.state_dict()
        model_keys = list(model_dict.keys())
        
        for key in checkpoint["model_state_dict"]:
            if key.startswith("module."):
                newkey = key.replace('module.', 'se3_Grid.')
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
            elif "tranistion." in key:
                newkey = key.replace('tranistion.', 'transition.')
                trained_dict[newkey] = checkpoint["model_state_dict"][key]
            else:
                if key in model_keys:
                    wts = checkpoint["model_state_dict"][key]
                    if wts.shape == model_dict[key].shape:
                        trained_dict[key] = wts
                    else:
                        print("skip", key)
        
        nnew, nexist = 0, 0
        for key in model_keys:
            if key not in trained_dict:
                nnew += 1
                print("new", key)
            else:
                nexist += 1
        
        if rank == 0:
            print("params", nnew, nexist)
        
        model.load_state_dict(trained_dict, strict=False)
        
        epoch = checkpoint["epoch"] + 1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]
        
        # Ensure all loss keys exist
        for key in train_loss_empty:
            if key not in train_loss:
                train_loss[key] = []
        for key in valid_loss_empty:
            if key not in valid_loss:
                valid_loss[key] = []
        
        if not config.silent:
            print("Restarting at epoch", epoch)
    
    else:
        if not config.silent:
            print("Training a new model")
        train_loss = train_loss_empty
        valid_loss = valid_loss_empty
        
        # Create model directory
        model_dir = join("models", config.modelname)
        if not isdir(model_dir):
            if not config.silent:
                print("Creating a new dir at", model_dir)
            os.makedirs(model_dir, exist_ok=True)
    
    # Re-initialize classification layers
    if epoch == 0:
        for i, (name, layer) in enumerate(model.named_modules()):
            if hasattr(layer, 'weight') and layer.weight is not None:
                print(name, layer.weight.data.numel())
            
            if isinstance(layer, torch.nn.Linear) and \
               ("class" in name or 'Xform' in name):
                if rank == 0:
                    print("reweight", name)
                layer.weight.data *= 0.1
    
    if rank == 0:
        print("Nparams:", count_parameters(model))
        print("Loaded")
    
    return model, optimizer, epoch, train_loss, valid_loss


def load_data(txt_file, world_size, rank, config):
    """Load dataset"""
    from torch.utils import data
    
    target_s = []
    ligands_s = []
    is_ligand_s = []
    decoy_npzs = []
    weights = {}
    
    print(txt_file)
    a = 0 
    with open(txt_file, 'r') as f:
        for ln in f:
            if ln.startswith('#'):
                continue
            a += 1
            x = ln.strip().split()
            is_ligand = bool(int(x[0]))  # 0:PP; 1:PL
            target = x[1]
            mol2type = x[2]  # how to read mol2f: single/batch
            mol2f = x[3]  # .npz or .mol2
            activemol = x[4]  # active molecule name or selection logic
            decoyf = x[5]  # .npz or batch-mol2
            
            if len(x) > 6:
                weights[target.split('/')[-1]] = float(x[6])
            else:
                weights[target.split('/')[-1]] = 1.0
            
            is_ligand_s.append(is_ligand)
            target_s.append(target)
            ligands_s.append((mol2type, mol2f, activemol, decoyf))
            
            if decoyf.endswith('.npz') and decoyf not in decoy_npzs:
                decoy_npzs.append(decoyf)
            if a == 50: 
                break 
    # Create dataset
    dataset_params = create_dataset_params(config)
    data_set = DataSet(target_s, is_ligand_s, ligands_s, decoy_npzs=decoy_npzs, **dataset_params)
    
    # Create dataloader
    dataloader_params = create_dataloader_params(config)
    
    if config.ddp:
        sampler = data.distributed.DistributedSampler(data_set, num_replicas=world_size, rank=rank)
        data_loader = data.DataLoader(data_set, sampler=sampler, **dataloader_params)
    else:
        data_loader = data.DataLoader(data_set, **dataloader_params)
    
    return data_loader, weights


def train_one_epoch(model, optimizer, loader, rank, epoch, is_train, config, weights):
    """Train for one epoch"""
    temp_loss = {
        "total": [], "BCEc": [], "BCEg": [], "BCEr": [], "contrast": [], 
        "reg": [], "struct": [], "mae": [], "spread": [], "Screen": [], 
        "ScreenC": [], "ScreenR": [], "Dkey": []
    }
    
    b_count, e_count = 0, 0
    accum = 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    Pt = {'CHEMBL': [], 'DUDE': [], 'PDB': []}
    Pf = {'CHEMBL': [], 'DUDE': [], 'PDB': []}
    
    for i, inputs in enumerate(loader):
        if inputs is None:
            e_count += 1
            continue
        
        (Grec, Glig, cats, masks, keyxyz, keyidx, blabel, info) = inputs
        if Grec is None:
            e_count += 1
            continue
        
        with torch.cuda.amp.autocast(enabled=False):
            with model.no_sync():
                t0 = time.time()
                
                # Move inputs to device
                if Glig is not None:
                    Glig = to_cuda(Glig, device)
                    keyxyz = to_cuda(keyxyz, device)
                    keyidx = to_cuda(keyidx, device)
                    nK = info['nK'].to(device)
                    blabel = to_cuda(blabel, device)
                else:
                    keyxyz, keyidx, nK, blabel = None, None, None, None
                
                Grec = to_cuda(Grec, device)
                pnames = info["pname"]
                grid = info['grid'].to(device)
                eval_struct = info['eval_struct'][0]
                grididx = info['grididx'].to(device)
                
                # Forward pass
                t1 = time.time()
                Yrec_s, Dkey_s, z, MotifP, aff = model(
                    Grec, Glig, keyidx, grididx,
                    gradient_checkpoint=is_train,
                    drop_out=is_train
                )
                
                if MotifP is None:
                    continue
                
                # Initialize losses
                lossGc = torch.tensor(0.0, device=device)
                lossGg = torch.tensor(0.0, device=device)
                lossGr = torch.tensor(0.0, device=device)
                lossGcontrast = torch.tensor(0.0, device=device)
                p_reg = torch.tensor(0.0, device=device)
                
                # GridNet losses
                if cats is not None:
                    cats = to_cuda(cats, device)
                    masks = to_cuda(masks, device)
                    
                    MotifP = torch.sigmoid(MotifP)
                    MotifPs = [MotifP]  # assume batchsize=1
                    
                    lossGc, lossGg, lossGr, bygrid = Loss.MaskedBCE(cats, MotifPs, masks)
                    lossGr = config.w_false * lossGr
                    lossGcontrast = config.w_contrast * Loss.ContrastLoss(MotifPs, masks)
                    p_reg = torch.nn.functional.relu(torch.sum(MotifP * MotifP - 25.0))
                
                # TR losses
                Pbind = []
                lossTs = torch.tensor(0.0, device=device)
                mae = torch.tensor(0.0, device=device)
                lossTr = torch.tensor(0.0, device=device)
                lossTd = torch.tensor(0.0, device=device)
                lossScreen = torch.tensor(0.0, device=device)
                lossScreenC = torch.tensor(0.0, device=device)
                lossScreenR = torch.tensor(0.0, device=device)
                
                if Yrec_s is not None and grid.shape[1] == z.shape[1]:
                    try:
                        # Structural loss
                        if len(nK.shape) > 1:
                            nK = nK.squeeze(dim=0)
                        
                        if eval_struct:
                            lossTs, mae = Loss.structural_loss(Yrec_s, keyxyz, nK, opt=config.struct_loss)
                            lossTd = config.w_Dkey * Loss.distance_loss2(Dkey_s, keyxyz, nK)
                            
                            lossTr = config.w_spread * Loss.spread_loss(keyxyz, z, grid, nK)
                            lossTr2 = config.w_spread * Loss.spread_loss2(keyxyz, z, grid, nK)
                            lossTr = lossTr + 0.2 * lossTr2
                        
                        # Screening loss
                        lossScreen = Loss.ScreeningLoss(aff[0], blabel)
                        lossScreenR = Loss.RankingLoss(torch.sigmoid(aff[0]), blabel)
                        lossScreenC = Loss.ScreeningContrastLoss(aff[1], blabel, nK)
                        Pbind = ['%4.2f' % float(a) for a in torch.sigmoid(aff[0])]
                        
                        # Determine dataset type
                        key = 'PDB'
                        if info['ligands'][0][0].startswith('CHEMBL') and pnames[0][0] in ['O', 'P', 'Q']:
                            key = 'CHEMBL'
                        elif info['ligands'][0][0].startswith('CHEMBL'):
                            key = 'DUDE'
                        
                        Pt[key].append(float(torch.sigmoid(aff[0][0]).cpu()))
                        Pf[key] += list(torch.sigmoid(aff[0][1:]).cpu().detach().numpy())
                    
                    except Exception as e:
                        print(f"Error in loss calculation: {e}")
                        pass
                
                # L2 regularization
                l2_reg = torch.tensor(0.0, device=device)
                if is_train:
                    for param in model.parameters():
                        l2_reg += torch.norm(param)
                
                # Final loss
                loss = (config.wGrid * (lossGc + lossGg + lossGr + lossGcontrast + 
                                       config.w_reg * (l2_reg + p_reg)) +
                        config.wTR * (lossTs + lossTr + lossTd) +
                        config.w_screen * lossScreen +
                        config.w_screen_ranking * lossScreenR +
                        config.w_screen_contrast * lossScreenC)
                
                # Apply target weight
                trg_weight = weights.get(pnames[0], 1.0)
                loss = loss * trg_weight
                
                # Store losses
                temp_loss["total"].append(loss.cpu().detach().numpy())
                temp_loss["BCEc"].append(lossGc.cpu().detach().numpy())
                temp_loss["BCEg"].append(lossGg.cpu().detach().numpy())
                temp_loss["BCEr"].append(lossGr.cpu().detach().numpy())
                temp_loss["contrast"].append(lossGcontrast.cpu().detach().numpy())
                temp_loss["reg"].append((p_reg + l2_reg).cpu().detach().numpy())
                
                if lossTs > 0.0:
                    temp_loss["struct"].append(lossTs.cpu().detach().numpy())
                    temp_loss["mae"].append(mae.cpu().detach().numpy())
                    temp_loss["spread"].append(lossTr.cpu().detach().numpy())
                    temp_loss["Dkey"].append(lossTd.cpu().detach().numpy())
                
                if lossScreen > 0.0:
                    temp_loss["Screen"].append(lossScreen.cpu().detach().numpy())
                    temp_loss["ScreenR"].append(lossScreenR.cpu().detach().numpy())
                    temp_loss["ScreenC"].append(lossScreenC.cpu().detach().numpy())
                
                # Backward pass and optimization
                gridinfo = info['gridinfo'][0].split('/')[-1].replace('.grid.npz', '')
                if is_train and (b_count + 1) % accum == 0:
                    loss.requires_grad_(True)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    print(f"Rank {rank} TRAIN Epoch({config.modelname}): [{epoch:2d}/{config.max_epoch:2d}], "
                          f"Batch: [{b_count:2d}/{len(loader):2d}], loss: {np.sum(temp_loss['total'][-accum:]):8.3f} "
                          f"(M/R/C/S(mae)/Sp/Scr: {np.sum(temp_loss['BCEc'][-accum:]):6.1f}/"
                          f"{np.sum(temp_loss['BCEr'][-accum:]):6.1f}/{np.sum(temp_loss['contrast'][-accum:]):6.3f}|"
                          f"{float(lossTs):6.1f}({float(mae):5.2f})/{float(lossTr):6.1f}/{float(lossTd):6.1f}|"
                          f"{np.sum(temp_loss['Screen'][-accum:]):4.2f}/{np.sum(temp_loss['ScreenR'][-accum:]):4.2f}), "
                          f"{pnames[0]}:{gridinfo}", ' '.join(Pbind), ','.join(info['ligands'][0]))
                
                elif (b_count + 1) % accum == 0:  # validation
                    print(f"Rank {rank} VALID Epoch({config.modelname}): [{epoch:2d}/{config.max_epoch:2d}], "
                          f"Batch: [{b_count:2d}/{len(loader):2d}], loss: {np.sum(temp_loss['total'][-accum:]):8.3f} "
                          f"(M/R/C|S(mae)/Sp/D|Scr: {np.sum(temp_loss['BCEc'][-accum:]):6.1f}/"
                          f"{np.sum(temp_loss['BCEr'][-accum:]):6.1f}/{np.sum(temp_loss['contrast'][-accum:]):6.3f}|"
                          f"{float(lossTs):6.1f}({float(mae):5.2f})/{float(lossTr):6.1f}/{float(lossTd):6.1f}|"
                          f"{np.sum(temp_loss['Screen'][-accum:]):4.2f}/{np.sum(temp_loss['ScreenR'][-accum:]):4.2f}), "
                          f"{pnames[0]}:{gridinfo}", ' '.join(Pbind), ','.join(info['ligands'][0]))
                
                b_count += 1
    
    return temp_loss, Pt, Pf


def train_model(rank, world_size, config):
    """Main training function"""
    gpu = rank % world_size
    dist.init_process_group(backend='gloo', world_size=world_size, rank=rank)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # Load model and parameters
    model, optimizer, start_epoch, train_loss, valid_loss = load_params(rank, config)
    
    if config.ddp:
        ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
    
    # Load data
    train_loader, weights_train = load_data(config.datasetf[0], world_size, rank, config)
    valid_loader, weights_valid = load_data(config.datasetf[1], world_size, rank, config)
    
    auc_train = {'CHEMBL': [], 'DUDE': [], 'PDB': []}
    auc_valid = {'CHEMBL': [], 'DUDE': [], 'PDB': []}
    
    # Training loop
    for epoch in range(start_epoch, config.max_epoch):
        # Train
        if config.ddp:
            ddp_model.train()
            temp_loss, Pt, Pf = train_one_epoch(ddp_model, optimizer, train_loader, rank, epoch, True, config, weights_train)
        else:
            model.train()
            temp_loss, Pt, Pf = train_one_epoch(model, optimizer, train_loader, rank, epoch, True, config, weights_train)
        
        for k in train_loss:
            train_loss[k].append(np.array(temp_loss[k]))
        
        if rank == 0:
            for key in Pt:
                if len(Pt[key]) > 10 and len(Pf[key]) > 10:
                    auc_train[key].append(calc_AUC(Pt[key], Pf[key]))
        
        optimizer.zero_grad()
        
        # Validation
        with torch.no_grad():
            if config.ddp:
                ddp_model.eval()
                temp_loss, Pt, Pf = train_one_epoch(ddp_model, optimizer, valid_loader, rank, epoch, False, config, weights_valid)
            else:
                temp_loss, Pt, Pf = train_one_epoch(model, optimizer, valid_loader, rank, epoch, False, config, weights_valid)
        
        for k in valid_loss:
            valid_loss[k].append(np.array(temp_loss[k]))
        
        if rank == 0:
            for key in Pt:
                if len(Pt[key]) > 10 and len(Pf[key]) > 10:
                    auc_valid[key].append(calc_AUC(Pt[key], Pf[key]))
        
        print("***SUM***")
        print(f"Train loss | {np.mean(train_loss['total'][-1]):7.4f} | Valid loss | {np.mean(valid_loss['total'][-1]):7.4f}")
        
        if rank == 0:
            auc_l = "AUC: "
            for key in ['PDB', 'CHEMBL', 'DUDE']:
                if key in auc_train and len(auc_train[key]) > 0:
                    auc_l += f' {key} {auc_train[key][-1]:6.4f}'
                if key in auc_valid and len(auc_valid[key]) > 0:
                    auc_l += f' / {auc_valid[key][-1]:6.4f}'
            print(auc_l)
        
        # Save checkpoints
        if rank == 0:
            model_dir = join("models", config.modelname)
            
            # Save best model
            if np.min([np.mean(vl) for vl in valid_loss["total"]]) == np.mean(valid_loss["total"][-1]):
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'auc_train': auc_train,
                    'auc_valid': auc_valid,
                }, join(model_dir, "best.pkl"))
            
            # Save current model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'auc_train': auc_train,
                'auc_valid': auc_valid,
            }, join(model_dir, "model.pkl"))
            
            # Save epoch-specific model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'auc_train': auc_train,
                'auc_valid': auc_valid,
            }, join(model_dir, f"epoch{epoch}.pkl"))


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train MotifScreen-Aff model')
    parser.add_argument('--config', type=str, default='common', 
                        help='Config name (e.g., common) or path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config.endswith('.yaml'):
        # Direct path to config file
        from configs.config import load_config
        config = load_config(args.config)
    else:
        # Config name - use predefined configs
        if args.config == 'common':
            config = create_common_config()
        else:
            # Try to load from YAML
            try:
                config = load_config_with_base(args.config)
            except FileNotFoundError:
                print(f"Config '{args.config}' not found. Using default common config.")
                config = create_common_config()
    
    # Override debug mode if specified
    if args.debug:
        config.debug = True
        config.dataloader.num_workers = 1
    
    print(f"DGL version: {dgl.__version__}")
    print(f"Using config: {config.modelname}")
    print(f"Model parameters: {config.m}D, dropout: {config.dropout_rate}")
    
    torch.cuda.empty_cache()
    mp.freeze_support()
    
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs..")
    
    # Set environment variables for distributed training
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12346'
    
    os.system(f"touch GPU {world_size}")
    
    if config.ddp:
        mp.spawn(train_model, args=(world_size, config), nprocs=world_size, join=True)
    else:
        train_model(0, 1, config)


if __name__ == "__main__":
    main()