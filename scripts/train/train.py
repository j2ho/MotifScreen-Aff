#!/usr/bin/env python
"""
Clean training script for MotifScreen-Aff using grouped parameter configuration.
Uses the new TrainingDataSet with organized parameter structure.
"""
import os
import sys
import numpy as np
from os.path import join, isdir
import torch
import time
import dgl
import argparse
import contextlib

# DDP related modules
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.data.training_dataset import TrainingDataSet, TrainingConfig, DataPaths, GraphConfig, DataProcessing, DataAugmentation, CrossValidation, collate
# from src.data.dataset_debug import DebugTrainingDataSet as TrainingDataSet
from src.model.models.msk1 import EndtoEndModel as MSK_1
from src.model.models.msk_v2 import EndtoEndModel as MSK_2

from scripts.train.utils import count_parameters, to_cuda, calc_AUC
import src.model.loss.losses as Loss
from configs.config import load_config_with_base

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")

import wandb


def create_training_config(config) -> TrainingConfig:
    paths = DataPaths(
        datapath=config.data.datapath,
        keyatomf=config.data.keyatomf,
        affinityf=config.data.affinityf,
        decoyf=config.data.decoyf,
    )

    graph = GraphConfig(
        edgemode=config.data.edgemode,
        edgek=tuple(config.data.edgek),
        edgedist=tuple(config.data.edgedist),
        maxedge=config.data.maxedge,
        maxnode=config.data.maxnode,
        ball_radius=config.data.ball_radius,
        firstshell_as_grid=config.firstshell_as_grid
    )

    processing = DataProcessing(
        ntype=config.data.ntype,
        max_subset=config.data.max_subset,
        drop_H=config.data.drop_H,
        store_memory=False  # Could be configurable
    )

    augmentation = DataAugmentation(
        randomize=config.data.randomize,
        randomize_grid=config.randomize_grid,  # Could be configurable
        pert=config.pert           # Could be configurable
    )

    cross_validation = CrossValidation(
        load_cross=config.load_cross,
        cross_eval_struct=config.cross_eval_struct,
        cross_grid=config.cross_grid,
        nonnative_struct_weight=config.nonnative_struct_weight
    )

    return TrainingConfig(
        paths=paths,
        graph=graph,
        processing=processing,
        augmentation=augmentation,
        cross_validation=cross_validation,
        mode='train',
        debug=config.debug
    )


def load_params(rank, config):
    """Load model, optimizer, and training state"""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Create model
    if config.version == "v1.0":
        if not config.silent:
            print("Loading MSK_1 model")
        model = MSK_1(config)
    elif config.version == "v2.0":
        if not config.silent:
            print("Loading MSK_2 model")
        model = MSK_2(config)
    model.to(device)

    # Initialize loss tracking
    train_loss_empty = {
        "total": [],
        "CatP": [], "CatN": [], "CatCont": [],
        "NormPenalty": [],
        "Str": [], "StrMAE": [], "StrPair": [], "KeyatmAttmap": [],
        "Screen": [], "ScreenC": [], "ScreenR": []
        }
    valid_loss_empty = {
        "total": [],
        "CatP": [], "CatN": [], "CatCont": [],
        "NormPenalty": [],
        "Str": [], "StrMAE": [], "StrPair": [], "KeyatmAttmap": [],
        "Screen": [], "ScreenC": [], "ScreenR": []
        }
    epoch = 0

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    # Load checkpoint if it exists
    checkpoint_path = join("models", f"{config.modelname}{config.version}", "model.pkl")
    if os.path.exists(checkpoint_path) and config.load_checkpoint:
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

        # if rank == 0:
        #     print("params", nnew, nexist)

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
        model_dir = join("models", f"{config.modelname}{config.version}")
        if not isdir(model_dir):
            if not config.silent:
                print("Creating a new dir at", model_dir)
            os.makedirs(model_dir, exist_ok=True)

    # Re-initialize classification layers
    if epoch == 0:
        for i, (name, layer) in enumerate(model.named_modules()):
            # if hasattr(layer, 'weight') and layer.weight is not None:
            #     print(name, layer.weight.data.numel())

            if isinstance(layer, torch.nn.Linear) and \
               ("class" in name or 'Xform' in name):
                # if rank == 0:
                #     print("reweight", name)
                layer.weight.data *= 0.1

    if rank == 0:
        print("Nparams:", count_parameters(model))
        print("Loaded")

    return model, optimizer, epoch, train_loss, valid_loss


def load_data(txt_file, world_size, rank, main_config):
    """Load dataset using grouped configuration"""
    from torch.utils import data

    # Create grouped training configuration
    training_data_config = create_training_config(main_config)

    # Parse training data file
    targets = []
    ligands = []
    weights = {}

    print(f"Loading training data from: {txt_file}")
    print(f"Graph config: edgemode={training_data_config.graph.edgemode}, "
          f"edgek={training_data_config.graph.edgek}, "
          f"ball_radius={training_data_config.graph.ball_radius}")
    print(f"Processing config: ntype={training_data_config.processing.ntype}, "
          f"max_subset={training_data_config.processing.max_subset}, "
          f"drop_H={training_data_config.processing.drop_H}")
    print(f"Augmentation config: randomize={training_data_config.augmentation.randomize}")

    with open(txt_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            parts = line.strip().split()

            target = parts[0]
            active_ligand = parts[1]
            mol2_file_type = parts[2]
            # Optional weight
            if len(parts) > 3:
                weights[target.split('/')[-1]] = float(parts[3])
            else:
                weights[target.split('/')[-1]] = 1.0

            targets.append(target)
            ligands.append((active_ligand, mol2_file_type))

    print(f"Loaded {len(targets)} samples from {txt_file}")

    # Create dataset with grouped configuration
    dataset = TrainingDataSet(
        targets=targets,
        ligands=ligands,
        config=training_data_config
    )

    # Create dataloader with standard parameters
    dataloader_params = {
        'shuffle': main_config.dataloader.shuffle,
        'num_workers': main_config.dataloader.num_workers,
        'pin_memory': main_config.dataloader.pin_memory,
        'collate_fn': collate,
        'batch_size': main_config.dataloader.batch_size
    }

    if main_config.ddp:
        sampler = data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = data.DataLoader(dataset, sampler=sampler, **dataloader_params)
    else:
        dataloader = data.DataLoader(dataset, **dataloader_params)

    return dataloader, weights


def train_one_epoch(model, optimizer, loader, rank, epoch, is_train, config, weights, global_step=0):
    """Train for one epoch"""
    temp_loss = {
        "total": [],
        "CatP": [], "CatN": [], "CatCont": [],
        "NormPenalty": [],
        "Str": [], "StrMAE": [], "StrPair": [], "KeyatmAttmap": [],
        "Screen": [], "ScreenC": [], "ScreenR": []
        }

    b_count, e_count = 0, 0
    accum = config.accumulation_steps
    if config.debug:
        accum = 1  # For debugging, use single step accumulation
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    Pt = {'chembl': [], 'biolip': [], 'pdbbind': []}
    Pf = {'chembl': [], 'biolip': [], 'pdbbind': []}

    for i, inputs in enumerate(loader):
        if inputs is None:
            e_count += 1
            continue

        (Grec, Glig, cats, masks, keyxyz, keyidx, blabel, info) = inputs
        grididx = info['grididx']
        if any(x is None for x in (Grec, Glig, keyidx, grididx)):
            e_count += 1
            continue

        with torch.cuda.amp.autocast(enabled=False):
            # Use no_sync only if model has it (DDP models), otherwise use nullcontext
            sync_context = model.no_sync() if hasattr(model, 'no_sync') else contextlib.nullcontext()
            with sync_context:
                t0 = time.time()

                # Move inputs to device
                Glig = to_cuda(Glig, device)
                keyxyz = to_cuda(keyxyz, device)
                keyidx = to_cuda(keyidx, device)
                nK = info['nK'].to(device)
                blabel = to_cuda(blabel, device)

                Grec = to_cuda(Grec, device)
                pnames = info["pname"]
                source = info['source'][0]
                grid = info['grid'].to(device)
                eval_struct = info['eval_struct'][0]
                grididx = grididx.to(device)
                # Forward pass
                t1 = time.time()
                keyxyz_pred, key_pairdist_pred, rec_key_z, motif_pred, bind_pred, absaff_pred = model(
                    Grec, Glig, keyidx, grididx,
                    gradient_checkpoint=is_train,
                    drop_out=is_train
                )

                if motif_pred is None:
                    continue

                # Initialize losses
                # Grid Motif Classification
                l_cat_pos = torch.tensor(0.0, device=device)
                l_cat_neg = torch.tensor(0.0, device=device)
                l_cat_contrast = torch.tensor(0.0, device=device)
                motif_penalty = torch.tensor(0.0, device=device)
                if cats is not None:
                    cats = to_cuda(cats, device)
                    masks = to_cuda(masks, device)

                    motif_pred = torch.sigmoid(motif_pred)
                    motif_preds = [motif_pred]  # assume batchsize=1

                    l_cat_pos, l_cat_neg = Loss.MaskedBCE(cats, motif_preds, masks)
                    l_cat_contrast = config.w_contrast * Loss.ContrastLoss(motif_preds, masks)
                    motif_penalty = torch.nn.functional.relu(torch.sum(motif_pred * motif_pred - 25.0))

                # Structure and Screening
                Pbind = []
                l_str_dist = torch.tensor(0.0, device=device)
                key_mae = torch.tensor(0.0, device=device)
                l_str_pair = torch.tensor(0.0, device=device)
                l_str_attmap = torch.tensor(0.0, device=device)
                l_screen = torch.tensor(0.0, device=device)
                l_screen_cont = torch.tensor(0.0, device=device)
                l_screen_rank = torch.tensor(0.0, device=device)

                if keyxyz_pred is not None and grid.shape[1] == rec_key_z.shape[1]:
                    try:
                        # Structural loss
                        if len(nK.shape) > 1:
                            nK = nK.squeeze(dim=0)

                        if eval_struct:
                            l_str_dist, key_mae = Loss.StructureLoss(keyxyz_pred, keyxyz, nK, opt=config.struct_loss)
                            l_str_pair = config.w_Dkey * Loss.PairDistanceLoss(key_pairdist_pred, keyxyz, nK)

                            l_str_attmap_pos= config.w_spread * Loss.SpreadLoss(keyxyz, rec_key_z, grid, nK)
                            l_str_attmap_neg = config.w_spread * Loss.SpreadLoss_v2(keyxyz, rec_key_z, grid, nK)
                            l_str_attmap = l_str_attmap_pos + 0.2 *l_str_attmap_neg
                        # Screening loss
                        l_screen = Loss.ScreeningLoss(bind_pred[0], blabel) # BCE with logits
                        l_screen_rank = Loss.RankingLoss(torch.sigmoid(bind_pred[0]), blabel)
                        l_screen_cont = Loss.ScreeningContrastLoss(bind_pred[1], blabel, nK)
                        Pbind = ['%4.2f' % float(a) for a in torch.sigmoid(bind_pred[0])]
                        Pt[source].append(float(torch.sigmoid(bind_pred[0][0]).cpu()))
                        Pf[source] += list(torch.sigmoid(bind_pred[0][1:]).cpu().detach().numpy())

                    except Exception as e:
                        print(f"Error in loss calculation: {e}")
                        # print traceback
                        import traceback
                        traceback.print_exc()
                        pass

                # L2 regularization
                l2_penalty = torch.tensor(0.0, device=device)
                if is_train:
                    for param in model.parameters():
                        l2_penalty += torch.norm(param)

                # Final loss
                loss = (config.w_cat * (l_cat_pos + l_cat_neg + l_cat_contrast +
                                       config.w_penalty * (l2_penalty + motif_penalty)) +
                        config.w_str * (l_str_dist + l_str_pair + l_str_attmap) +
                        config.w_screen * l_screen +
                        config.w_screen_ranking * l_screen_rank +
                        config.w_screen_contrast * l_screen_cont)

                # Apply target weight
                trg_weight = weights.get(pnames[0], 1.0)
                loss = loss * trg_weight

                # Store losses
                temp_loss["total"].append(loss.cpu().detach().numpy())
                temp_loss["CatP"].append(l_cat_pos.cpu().detach().numpy())
                temp_loss["CatN"].append(l_cat_neg.cpu().detach().numpy())
                temp_loss["CatCont"].append(l_cat_contrast.cpu().detach().numpy())
                temp_loss["NormPenalty"].append((motif_penalty + l2_penalty).cpu().detach().numpy())

                if l_str_dist > 0.0:
                    temp_loss["Str"].append(l_str_dist.cpu().detach().numpy())
                    temp_loss["StrMAE"].append(key_mae.cpu().detach().numpy())
                    temp_loss["StrPair"].append(l_str_pair.cpu().detach().numpy())
                    temp_loss["KeyatmAttmap"].append(l_str_attmap.cpu().detach().numpy())

                if l_screen > 0.0:
                    temp_loss["Screen"].append(l_screen.cpu().detach().numpy())
                    temp_loss["ScreenR"].append(l_screen_rank.cpu().detach().numpy())
                    temp_loss["ScreenC"].append(l_screen_cont.cpu().detach().numpy())

                # Backward pass and optimization
                if is_train and (b_count + 1) % accum == 0:
                    loss.requires_grad_(True)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    # Log to wandb at step level for training (rank 0 only)
                    if rank == 0 and not config.debug:
                        step_metrics = {
                            "step": global_step + b_count,
                            "train_step/loss_total": float(loss.cpu().detach().numpy()),
                            "train_step/loss_cat_pos": float(l_cat_pos.cpu().detach().numpy()),
                            "train_step/loss_cat_neg": float(l_cat_neg.cpu().detach().numpy()),
                            "train_step/loss_cat_contrast": float(l_cat_contrast.cpu().detach().numpy()),
                            "train_step/loss_norm_penalty": float((motif_penalty + l2_penalty).cpu().detach().numpy())
                        }

                        if l_str_dist > 0.0:
                            step_metrics.update({
                                "train_step/loss_structure": float(l_str_dist.cpu().detach().numpy()),
                                "train_step/loss_structure_mae": float(key_mae.cpu().detach().numpy()),
                                "train_step/loss_structure_pair": float(l_str_pair.cpu().detach().numpy()),
                                "train_step/loss_keyatm_attmap": float(l_str_attmap.cpu().detach().numpy())
                            })

                        if l_screen > 0.0:
                            step_metrics.update({
                                "train_step/loss_screening": float(l_screen.cpu().detach().numpy()),
                                "train_step/loss_screening_rank": float(l_screen_rank.cpu().detach().numpy()),
                                "train_step/loss_screening_contrast": float(l_screen_cont.cpu().detach().numpy())
                            })

                        wandb.log(step_metrics)

                #print losses
                    print (f"Rank {rank} TRAIN Epoch: [{epoch:2d}/{config.max_epoch:2d}], "
                          f"Batch: [{b_count:2d}/{len(loader):2d}], loss: {np.sum(temp_loss['total'][-accum:]):8.3f} "
                          f"(Category Pos/Neg/Contrast: {np.sum(temp_loss['CatP'][-accum:]):6.1f}/"
                          f"{np.sum(temp_loss['CatN'][-accum:]):6.1f}/{np.sum(temp_loss['CatCont'][-accum:]):6.3f}|"
                          f"Struct Dist/Mae/Pair: {np.sum(temp_loss['Str'][-accum:]):6.1f}/"
                          f"{np.sum(temp_loss['StrMAE'][-accum:]):5.2f}/"
                          f"{np.sum(temp_loss['StrPair'][-accum:]):6.3f}|"
                          f"Screening bce/rank/Contrast: {np.sum(temp_loss['Screen'][-accum:]):4.2f}/"
                          f"{np.sum(temp_loss['ScreenR'][-accum:]):4.2f}/"
                          f"{np.sum(temp_loss['ScreenC'][-accum:]):6.3f}|"
                          f"{pnames[0]}:", ' '.join(Pbind), ','.join(info['ligands'][0]))

                elif (b_count + 1) % accum == 0:  # validation
                    # Log to wandb at step level for validation (rank 0 only)
                    if rank == 0 and not config.debug:
                        step_metrics = {
                            "step": global_step + b_count,
                            "valid_step/loss_total": float(loss.cpu().detach().numpy()),
                            "valid_step/loss_cat_pos": float(l_cat_pos.cpu().detach().numpy()),
                            "valid_step/loss_cat_neg": float(l_cat_neg.cpu().detach().numpy()),
                            "valid_step/loss_cat_contrast": float(l_cat_contrast.cpu().detach().numpy()),
                            "valid_step/loss_norm_penalty": float((motif_penalty + l2_penalty).cpu().detach().numpy())
                        }

                        if l_str_dist > 0.0:
                            step_metrics.update({
                                "valid_step/loss_structure": float(l_str_dist.cpu().detach().numpy()),
                                "valid_step/loss_structure_mae": float(key_mae.cpu().detach().numpy()),
                                "valid_step/loss_structure_pair": float(l_str_pair.cpu().detach().numpy()),
                                "valid_step/loss_keyatm_attmap": float(l_str_attmap.cpu().detach().numpy())
                            })

                        if l_screen > 0.0:
                            step_metrics.update({
                                "valid_step/loss_screening": float(l_screen.cpu().detach().numpy()),
                                "valid_step/loss_screening_rank": float(l_screen_rank.cpu().detach().numpy()),
                                "valid_step/loss_screening_contrast": float(l_screen_cont.cpu().detach().numpy())
                            })

                        wandb.log(step_metrics)

                    print (f"Rank {rank} VALID Epoch: [{epoch:2d}/{config.max_epoch:2d}], "
                          f"Batch: [{b_count:2d}/{len(loader):2d}], loss: {np.sum(temp_loss['total'][-accum:]):8.3f} "
                          f"(Category Pos/Neg/Contrast: {np.sum(temp_loss['CatP'][-accum:]):6.1f}/"
                          f"{np.sum(temp_loss['CatN'][-accum:]):6.1f}/{np.sum(temp_loss['CatCont'][-accum:]):6.3f}|"
                          f"Struct Dist/Mae/Pair: {np.sum(temp_loss['Str'][-accum:]):6.1f}/"
                          f"{np.sum(temp_loss['StrMAE'][-accum:]):5.2f}/"
                          f"{np.sum(temp_loss['StrPair'][-accum:]):6.3f}|"
                          f"Screening bce/rank/Contrast: {np.sum(temp_loss['Screen'][-accum:]):4.2f}/"
                          f"{np.sum(temp_loss['ScreenR'][-accum:]):4.2f}/"
                          f"{np.sum(temp_loss['ScreenC'][-accum:]):6.3f}|"
                          f"{pnames[0]}", ' '.join(Pbind), ','.join(info['ligands'][0]))

                b_count += 1

    return temp_loss, Pt, Pf


def train_model(rank, world_size, config):
    """Main training function"""
    gpu = rank % world_size
    dist.init_process_group(backend='gloo', world_size=world_size, rank=rank)

    # Initialize wandb for rank 0 only
    if rank == 0 and not config.debug:
        wandb.init(
            project="motifscreen-aff",
            name=f"{config.modelname}{config.version}_epoch{config.max_epoch}",
            mode="online",
            config={
                "model_version": config.version,
                "model_name": config.modelname,
                "learning_rate": config.LR,
                "max_epochs": config.max_epoch,
                "batch_size": config.dataloader.batch_size,
                "dropout_rate": config.dropout_rate,
                "edge_mode": config.data.edgemode,
                "ball_radius": config.data.ball_radius,
                "w_cat": config.w_cat,
                "w_str": config.w_str,
                "w_screen": config.w_screen
            }
        )

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Load model and parameters
    model, optimizer, start_epoch, train_loss, valid_loss = load_params(rank, config)

    if config.ddp:
        if torch.cuda.is_available():
            ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        else:
            ddp_model = DDP(model, find_unused_parameters=False)

    # Load data using grouped configuration
    if config.debug:
        train_datasetf = 'data/small.txt'
        valid_datasetf = 'data/small.txt'
    else:
        train_datasetf = config.datasetf[0]
        valid_datasetf = config.datasetf[1]
    train_loader, weights_train = load_data(train_datasetf, world_size, rank, config)
    valid_loader, weights_valid = load_data(valid_datasetf, world_size, rank, config)

    auc_train = {'chembl': [], 'biolip': [], 'pdbbind': []}
    auc_valid = {'chembl': [], 'biolip': [], 'pdbbind': []}

    # Initialize global step counter
    global_step = start_epoch * len(train_loader)

    # Training loop
    for epoch in range(start_epoch, config.max_epoch):
        print(f"\n=== Epoch {epoch}/{config.max_epoch} ===")

        # Train
        if config.ddp:
            ddp_model.train()
            temp_loss, Pt, Pf = train_one_epoch(ddp_model, optimizer, train_loader, rank, epoch, True, config, weights_train, global_step)
        else:
            model.train()
            temp_loss, Pt, Pf = train_one_epoch(model, optimizer, train_loader, rank, epoch, True, config, weights_train, global_step)

        for k in train_loss:
            train_loss[k].append(np.array(temp_loss[k]))

        if rank == 0:
            for key in Pt:
                if len(Pt[key]) > 10 and len(Pf[key]) > 10:
                    auc_train[key].append(calc_AUC(Pt[key], Pf[key]))

            # Log training losses to wandb
            if not config.debug:
                train_metrics = {
                    "epoch": epoch,
                    "train/loss_total": np.mean(train_loss['total'][-1]),
                    "train/loss_cat_pos": np.mean(train_loss['CatP'][-1]),
                    "train/loss_cat_neg": np.mean(train_loss['CatN'][-1]),
                    "train/loss_cat_contrast": np.mean(train_loss['CatCont'][-1]),
                    "train/loss_norm_penalty": np.mean(train_loss['NormPenalty'][-1])
                }

                if len(train_loss['Str'][-1]) > 0:
                    train_metrics.update({
                        "train/loss_structure": np.mean(train_loss['Str'][-1]),
                        "train/loss_structure_mae": np.mean(train_loss['StrMAE'][-1]),
                        "train/loss_structure_pair": np.mean(train_loss['StrPair'][-1]),
                        "train/loss_keyatm_attmap": np.mean(train_loss['KeyatmAttmap'][-1])
                    })

                if len(train_loss['Screen'][-1]) > 0:
                    train_metrics.update({
                        "train/loss_screening": np.mean(train_loss['Screen'][-1]),
                        "train/loss_screening_rank": np.mean(train_loss['ScreenR'][-1]),
                        "train/loss_screening_contrast": np.mean(train_loss['ScreenC'][-1])
                    })

                # Add AUC values
                for key in ['pdbbind', 'chembl', 'biolip']:
                    if key in auc_train and len(auc_train[key]) > 0:
                        train_metrics[f"train/auc_{key}"] = auc_train[key][-1]

                wandb.log(train_metrics)

        optimizer.zero_grad()

        # Update global step counter after training
        global_step += len(train_loader)

        # Validation
        with torch.no_grad():
            if config.ddp:
                ddp_model.eval()
                temp_loss, Pt, Pf = train_one_epoch(ddp_model, optimizer, valid_loader, rank, epoch, False, config, weights_valid, global_step)
            else:
                model.eval()
                temp_loss, Pt, Pf = train_one_epoch(model, optimizer, valid_loader, rank, epoch, False, config, weights_valid, global_step)

        for k in valid_loss:
            valid_loss[k].append(np.array(temp_loss[k]))

        if rank == 0:
            for key in Pt:
                if len(Pt[key]) > 10 and len(Pf[key]) > 10:
                    auc_valid[key].append(calc_AUC(Pt[key], Pf[key]))

            # Log validation losses to wandb
            if not config.debug:
                valid_metrics = {
                    "epoch": epoch,
                    "valid/loss_total": np.mean(valid_loss['total'][-1]),
                    "valid/loss_cat_pos": np.mean(valid_loss['CatP'][-1]),
                    "valid/loss_cat_neg": np.mean(valid_loss['CatN'][-1]),
                    "valid/loss_cat_contrast": np.mean(valid_loss['CatCont'][-1]),
                    "valid/loss_norm_penalty": np.mean(valid_loss['NormPenalty'][-1])
                }

                if len(valid_loss['Str'][-1]) > 0:
                    valid_metrics.update({
                        "valid/loss_structure": np.mean(valid_loss['Str'][-1]),
                        "valid/loss_structure_mae": np.mean(valid_loss['StrMAE'][-1]),
                        "valid/loss_structure_pair": np.mean(valid_loss['StrPair'][-1]),
                        "valid/loss_keyatm_attmap": np.mean(valid_loss['KeyatmAttmap'][-1])
                    })

                if len(valid_loss['Screen'][-1]) > 0:
                    valid_metrics.update({
                        "valid/loss_screening": np.mean(valid_loss['Screen'][-1]),
                        "valid/loss_screening_rank": np.mean(valid_loss['ScreenR'][-1]),
                        "valid/loss_screening_contrast": np.mean(valid_loss['ScreenC'][-1])
                    })

                # Add AUC values
                for key in ['pdbbind', 'chembl', 'biolip']:
                    if key in auc_valid and len(auc_valid[key]) > 0:
                        valid_metrics[f"valid/auc_{key}"] = auc_valid[key][-1]

                wandb.log(valid_metrics)

        print("***SUM***")
        print(f"Train loss | {np.mean(train_loss['total'][-1]):7.4f} | Valid loss | {np.mean(valid_loss['total'][-1]):7.4f}")

        if rank == 0:
            auc_l = "AUC: "
            for key in ['pdbbind', 'chembl', 'biolip']:
                if key in auc_train and len(auc_train[key]) > 0:
                    auc_l += f' {key} {auc_train[key][-1]:6.4f}'
                if key in auc_valid and len(auc_valid[key]) > 0:
                    auc_l += f' / {auc_valid[key][-1]:6.4f}'
            print(auc_l)

        # Save checkpoints
        if rank == 0:
            model_dir = join("models", f"{config.modelname}{config.version}")

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
                print("Saved best model")

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
    parser = argparse.ArgumentParser(description='Train MotifScreen-Aff')
    parser.add_argument('--config', type=str, default='common',
                        help='Config name (e.g., common) or path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    # Additional parameter override arguments
    parser.add_argument('--version', type=str, help='Override model version (e.g., v1.0, v2.0)')

    args = parser.parse_args()

    # Load configuration
    if args.config.endswith('.yaml'):
        # Direct path to config file
        from configs.config import load_config
        config = load_config(args.config)
    else:
        # Try to load from YAML
        try:
            config = load_config_with_base(args.config)
        except FileNotFoundError:
            print(f"Config '{args.config}' not found. Using default common config.")

    # Override debug mode if specified
    if args.debug:
        config.debug = True
        config.dataloader.num_workers = 1
    if args.version:
        config.version = args.version

    print(f"DGL version: {dgl.__version__}")
    print(f"Using config: {args.config}")
    print(f"Using model: MSK{config.version}")
    print(f"Training dropout: {config.dropout_rate}")

    # Print grouped parameter summary
    print("\n=== Grouped Parameter Configuration ===")
    print(f"Graph: edgemode={config.data.edgemode}, edgek={config.data.edgek}, "
          f"edgedist={config.data.edgedist}, ball_radius={config.data.ball_radius}")
    print(f"Processing: ntype={config.data.ntype}, max_subset={config.data.max_subset}, "
          f"drop_H={config.data.drop_H}")
    print(f"Augmentation: randomize={config.data.randomize}")
    print(f"Cross-validation: load_cross={config.load_cross}, "
          f"cross_eval_struct={config.cross_eval_struct}")
    print("="*50)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mp.freeze_support()

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Using {world_size} GPUs.." if torch.cuda.is_available() else "Using CPU only..")

    # Disable DDP when using CPU only
    if not torch.cuda.is_available():
        config.ddp = False
        print("Disabled DDP for CPU-only execution")

    # Set environment variables for distributed training
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12346'

    if config.ddp:
        mp.spawn(train_model, args=(world_size, config), nprocs=world_size, join=True)
    else:
        train_model(0, 1, config)


if __name__ == "__main__":
    main()
