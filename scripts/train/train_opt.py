
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

from src.data.dataset_opt import TrainingDataSet, collate
from src.model.models.msk1 import EndtoEndModel as MSK_1
from src.model.models.msk_ab import EndtoEndModel as MSK_AB

from scripts.train.utils import count_parameters, to_cuda, calc_AUC
import src.model.loss.losses as Loss
from configs.config_loader import load_config, load_config_with_base, Config

import warnings
warnings.filterwarnings("ignore", message="sourceTensor.clone")

import wandb

def load_params(rank, config: Config):
    """Load model, optimizer, and training state"""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    if config.version == "v1.0":
        if not config.training.silent:
            print("Loading MSK_1 model (default)")
        model = MSK_1(config)
    elif config.version == "ablation":
        if not config.training.silent:
            print("Loading MSK_1 model (ablation)")
        model = MSK_AB(config)
    model.to(device)

    train_loss_empty = {
        "total": [], "CatP": [], "CatN": [], "CatCont": [], "NormPenalty": [],
        "L2Penalty": [], "MotifPenalty": [],
        "Str": [], "StrMAE": [], "StrPair": [], "KeyatmAttmap": [],
        "Screen": [], "ScreenC": [], "ScreenR": []
        }
    valid_loss_empty = {
        "total": [], "CatP": [], "CatN": [], "CatCont": [], "NormPenalty": [],
        "L2Penalty": [], "MotifPenalty": [],
        "Str": [], "StrMAE": [], "StrPair": [], "KeyatmAttmap": [],
        "Screen": [], "ScreenC": [], "ScreenR": []
        }
    epoch = 0

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    
    scheduler = None
    if config.training.scheduler.use_scheduler:
        if config.training.scheduler.scheduler_type == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=config.training.scheduler.factor,
                patience=config.training.scheduler.patience,
                verbose=True,
                min_lr=config.training.scheduler.min_lr,
                threshold=config.training.scheduler.threshold
            )
        elif config.training.scheduler.scheduler_type == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config.training.scheduler.step_size,
                gamma=config.training.scheduler.gamma
            )
        elif config.training.scheduler.scheduler_type == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.training.scheduler.T_max
            )
        else:
            print(f"Unknown scheduler type: {config.training.scheduler.scheduler_type}. No scheduler will be used.")
    else:
        print("Scheduler disabled in config.")

    checkpoint_path = join("models", f"{config.modelname}{config.version}", "model.pkl")
    if os.path.exists(checkpoint_path) and config.training.load_checkpoint:
        if not config.training.silent:
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

        model.load_state_dict(trained_dict, strict=False)

        epoch = checkpoint["epoch"] + 1
        train_loss = checkpoint["train_loss"]
        valid_loss = checkpoint["valid_loss"]

        for key in train_loss_empty:
            if key not in train_loss:
                train_loss[key] = []
        for key in valid_loss_empty:
            if key not in valid_loss:
                valid_loss[key] = []

        if not config.training.silent:
            print("Restarting at epoch", epoch)

    else:
        if not config.training.silent:
            print("Training a new model")
        train_loss = train_loss_empty
        valid_loss = valid_loss_empty

        model_dir = join("models", f"{config.modelname}{config.version}")
        if not isdir(model_dir):
            if not config.training.silent:
                print("Creating a new dir at", model_dir)
            os.makedirs(model_dir, exist_ok=True)

    if epoch == 0:
        # Improved parameter initialization to prevent parameter explosion
        for i, (name, layer) in enumerate(model.named_modules()):
            if isinstance(layer, torch.nn.Linear):
                if "class" in name or 'Xform' in name:
                    # Scale down output layers more aggressively
                    layer.weight.data *= 0.1
                elif "trigon" in name.lower() or "attention" in name.lower():
                    # Special initialization for attention layers to prevent explosion
                    torch.nn.init.xavier_uniform_(layer.weight, gain=0.5)
                    if layer.bias is not None:
                        torch.nn.init.zeros_(layer.bias)
            elif isinstance(layer, torch.nn.LayerNorm):
                # Ensure LayerNorm is properly initialized
                if layer.weight is not None:
                    torch.nn.init.ones_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    if rank == 0:
        print("Nparams:", count_parameters(model))
        print("Loaded")

    return model, optimizer, scheduler, epoch, train_loss, valid_loss


def load_data(txt_file, world_size, rank, main_config: Config):
    """Load dataset using grouped configuration"""
    from torch.utils import data

    # Parse training data file
    targets = []
    ligands = []
    weights = {}

    print(f"Loading training data from: {txt_file}")
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

    dataset = TrainingDataSet(
        targets=targets,
        ligands=ligands,
        config=main_config
    )

    dataloader_params = {
        'shuffle': main_config.dataloader.shuffle,
        'num_workers': main_config.dataloader.num_workers,
        'pin_memory': main_config.dataloader.pin_memory,
        'collate_fn': collate,
        'batch_size': main_config.dataloader.batch_size
    }

    if main_config.training.ddp:
        sampler = data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        dataloader = data.DataLoader(dataset, sampler=sampler, **dataloader_params)
    else:
        dataloader = data.DataLoader(dataset, **dataloader_params)

    return dataloader, weights


class OptimizedLossTracker:
    """OPTIMIZED: Tracks losses on GPU without forcing synchronization"""
    def __init__(self, device):
        self.device = device
        self.reset()
    
    def reset(self):
        self.losses = {
            "total": [], "CatP": [], "CatN": [], "CatCont": [], "NormPenalty": [],
            "L2Penalty": [], "MotifPenalty": [],
            "Str": [], "StrMAE": [], "StrPair": [], "KeyatmAttmap": [],
            "Screen": [], "ScreenC": [], "ScreenR": []
        }
    
    def add_loss(self, loss_dict):
        """Add losses without CPU synchronization"""
        for key, value in loss_dict.items():
            if value is not None and torch.is_tensor(value):
                # Keep on GPU, only detach from computation graph
                self.losses[key].append(value.detach())
    
    def get_cpu_losses(self):
        """Convert accumulated GPU losses to CPU for logging (single sync point)"""
        cpu_losses = {}
        for key, values in self.losses.items():
            if values:
                # Single batch transfer to CPU
                stacked = torch.stack(values) if len(values) > 1 else values[0].unsqueeze(0)
                cpu_losses[key] = stacked.cpu().numpy()
            else:
                cpu_losses[key] = np.array([])
        return cpu_losses


def optimized_device_transfer(inputs, device):
    """OPTIMIZED: Single-pass device transfer for all inputs"""
    (Grec, Glig, cats, masks, keyxyz, keyidx, blabel, info) = inputs
    
    # Main graph objects
    if Glig is not None:
        Glig = Glig.to(device, non_blocking=True)
        # Ensure gdata (custom attribute) is also moved to device
        if hasattr(Glig, 'gdata'):
            Glig.gdata = Glig.gdata.to(device, non_blocking=True)
    if Grec is not None:
        Grec = Grec.to(device, non_blocking=True)
    
    # Tensor data - batch transfer
    tensor_data = {
        'keyxyz': keyxyz,
        'keyidx': keyidx, 
        'blabel': blabel,
        'nK': info['nK'],
        'grid': info['grid'],
        'grididx': info['grididx']
    }
    
    # Transfer tensors in batch
    for key, tensor in tensor_data.items():
        if tensor is not None:
            if isinstance(tensor, list):
                tensor_data[key] = [t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in tensor]
            elif torch.is_tensor(tensor):
                tensor_data[key] = tensor.to(device, non_blocking=True)
    
    # Transfer cats/masks if present
    if cats is not None:
        cats = cats.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
    
    return (Grec, Glig, cats, masks, 
            tensor_data['keyxyz'], tensor_data['keyidx'], tensor_data['blabel'],
            tensor_data['nK'], tensor_data['grid'], tensor_data['grididx'])


def train_one_epoch(model, optimizer, loader, rank, epoch, is_train, config: Config, weights, global_step=0):
    """OPTIMIZED: Train for one epoch with reduced CUDA synchronization"""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    # OPTIMIZED: Use GPU-based loss tracking
    loss_tracker = OptimizedLossTracker(device)
    
    b_count, e_count = 0, 0
    accum = config.training.accumulation_steps
    if config.training.debug:
        accum = 1

    Pt = {'chembl': [], 'biolip': [], 'pdbbind': []}
    Pf = {'chembl': [], 'biolip': [], 'pdbbind': []}

    for i, inputs in enumerate(loader):
        if inputs is None:
            e_count += 1
            continue

        # OPTIMIZED: Single device transfer operation
        try:
            (Grec, Glig, cats, masks, keyxyz, keyidx, blabel, nK, grid, grididx) = optimized_device_transfer(inputs, device)
            info = inputs[-1]  # Info dict stays on CPU
        except Exception as e:
            print(f"Device transfer failed: {e}")
            e_count += 1
            continue

        if any(x is None for x in (Grec, Glig, keyidx, grididx)):
            e_count += 1
            continue

        with torch.cuda.amp.autocast(config.training.amp):
            sync_context = model.no_sync() if hasattr(model, 'no_sync') else contextlib.nullcontext()
            with sync_context:
                t0 = time.time()

                pnames = info["pname"]
                source = info['source'][0]
                eval_struct = info['eval_struct'][0]
                t1 = time.time()
                keyxyz_pred, key_pairdist_pred, rec_key_z, motif_pred, bind_pred, absaff_pred = model(
                    Grec, Glig, keyidx, grididx,
                    gradient_checkpoint=is_train,
                    drop_out=is_train
                )

                if motif_pred is None:
                    continue

                # Early NaN detection after model forward pass
                nan_detected = False
                if keyxyz_pred is not None and torch.isnan(keyxyz_pred).any():
                    print(f"NaN detected in keyxyz_pred at epoch {epoch}, batch {b_count}")
                    nan_detected = True
                if key_pairdist_pred is not None and torch.isnan(key_pairdist_pred).any():
                    print(f"NaN detected in key_pairdist_pred at epoch {epoch}, batch {b_count}")
                    nan_detected = True
                if rec_key_z is not None and torch.isnan(rec_key_z).any():
                    print(f"NaN detected in rec_key_z at epoch {epoch}, batch {b_count}")
                    nan_detected = True
                if motif_pred is not None and torch.isnan(motif_pred).any():
                    print(f"NaN detected in motif_pred at epoch {epoch}, batch {b_count}")
                    nan_detected = True
                if bind_pred is not None:
                    for i, bp in enumerate(bind_pred):
                        if bp is not None and torch.isnan(bp).any():
                            print(f"NaN detected in bind_pred[{i}] at epoch {epoch}, batch {b_count}")
                            nan_detected = True
                
                if nan_detected:
                    print(f"Skipping batch {b_count} due to NaN in model outputs")
                    continue

                l_cat_pos = torch.tensor(0.0, device=device)
                l_cat_neg = torch.tensor(0.0, device=device)
                l_cat_contrast = torch.tensor(0.0, device=device)
                motif_penalty = torch.tensor(0.0, device=device)
                if cats is not None and eval_struct:
                    motif_pred = torch.sigmoid(motif_pred)
                    motif_preds = [motif_pred]

                    l_cat_pos, l_cat_neg = Loss.MaskedBCE(cats, motif_preds, masks)
                    l_cat_contrast = config.losses.w_contrast * Loss.ContrastLoss(motif_preds, masks)
                    motif_penalty = torch.nn.functional.relu(torch.sum(motif_pred * motif_pred - 25.0))

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
                        if len(nK.shape) > 1:
                            nK = nK.squeeze(dim=0)

                        if eval_struct:
                            # Access struct_loss from config.losses
                            l_str_dist, key_mae = Loss.StructureLoss(keyxyz_pred, keyxyz, nK, opt=config.losses.struct_loss)
                            # Access w_Dkey from config.losses
                            l_str_pair = config.losses.w_Dkey * Loss.PairDistanceLoss(key_pairdist_pred, keyxyz, nK)

                            # Access w_spread from config.losses
                            l_str_attmap_pos= config.losses.w_spread * Loss.SpreadLoss(keyxyz, rec_key_z, grid, nK)
                            l_str_attmap_neg = config.losses.w_spread * Loss.SpreadLoss_v2(keyxyz, rec_key_z, grid, nK)
                            l_str_attmap = l_str_attmap_pos + 0.2 *l_str_attmap_neg

                    except Exception as e:
                        print(f"Error in str loss calculation: {e}")
                        import traceback
                        traceback.print_exc()
                        pass
                if bind_pred is not None:
                    try:
                        # Access screening loss weights from config.losses
                        l_screen = Loss.ScreeningLoss(bind_pred[0], blabel) # BCEwithlogits, has sigmoid inside
                        l_screen_rank = Loss.RankingLoss(torch.sigmoid(bind_pred[0]), blabel)
                        l_screen_cont = Loss.ScreeningContrastLoss(bind_pred[1], blabel, nK)
                        Pbind = ['%4.2f' % float(a) for a in torch.sigmoid(bind_pred[0])]
                        Pt[source].append(float(torch.sigmoid(bind_pred[0][0]).cpu()))
                        Pf[source] += list(torch.sigmoid(bind_pred[0][1:]).cpu().detach().numpy())
                    except Exception as e:
                        print(f"Error in binding loss calculation: {e}")
                        import traceback
                        traceback.print_exc()
                        pass
                    
                l2_penalty = torch.tensor(0.0, device=device)
                if is_train:
                    for param in model.parameters():
                        l2_penalty += torch.norm(param)

                # Access all loss weights from config.losses
                loss = (config.losses.w_cat * (l_cat_pos + l_cat_neg + l_cat_contrast +
                                       config.losses.w_penalty * (l2_penalty + motif_penalty)) +
                        config.losses.w_str * (l_str_dist + l_str_pair + l_str_attmap) +
                        config.losses.w_screen * l_screen +
                        config.losses.w_screen_ranking * l_screen_rank +
                        config.losses.w_screen_contrast * l_screen_cont)

                trg_weight = weights.get(pnames[0], 1.0)
                loss = loss * trg_weight

                # Check for NaN/Inf in total loss before recording
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf detected in total loss at epoch {epoch}, batch {b_count}")
                    print(f"  Individual losses: cat_pos={l_cat_pos:.6f}, cat_neg={l_cat_neg:.6f}")
                    print(f"  cat_contrast={l_cat_contrast:.6f}, str_dist={l_str_dist:.6f}")
                    print(f"  screen={l_screen:.6f}, screen_rank={l_screen_rank:.6f}")
                    print(f"  motif_penalty={motif_penalty:.6f}, l2_penalty={l2_penalty:.6f}")
                    print(f"  target_weight={trg_weight:.6f}")
                    continue

                # OPTIMIZED: Add losses to GPU tracker (no CPU sync)
                loss_dict = {
                    "total": loss,
                    "CatP": l_cat_pos,
                    "CatN": l_cat_neg,
                    "CatCont": l_cat_contrast,
                    "NormPenalty": (motif_penalty + l2_penalty),
                    "L2Penalty": l2_penalty,
                    "MotifPenalty": motif_penalty
                }
                
                if l_str_dist > 0.0:
                    loss_dict.update({
                        "Str": l_str_dist,
                        "StrMAE": key_mae,
                        "StrPair": l_str_pair,
                        "KeyatmAttmap": l_str_attmap
                    })

                if l_screen > 0.0:
                    loss_dict.update({
                        "Screen": l_screen,
                        "ScreenR": l_screen_rank,
                        "ScreenC": l_screen_cont
                    })
                
                loss_tracker.add_loss(loss_dict)

                if is_train and (b_count + 1) % accum == 0:
                    loss.requires_grad_(True)
                    loss.backward()
                    
                    # Add gradient clipping to prevent NaN and monitor gradient norm
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Monitor parameter norms and clip if necessary
                    # param_norm = sum(p.norm().item() for p in model.parameters() if p.requires_grad)
                    
                    # Check for parameter explosion and apply clipping if needed
                    # if param_norm > config.training.max_param_norm:
                    #     print(f"WARNING: Parameter norm {param_norm:.2f} exceeds threshold {config.training.max_param_norm}")
                    #     # Clip parameters to prevent explosion
                    #     with torch.no_grad():
                    #         for param in model.parameters():
                    #             if param.requires_grad:
                    #                 param.data.clamp_(-10.0, 10.0)
                    #     param_norm = sum(p.norm().item() for p in model.parameters() if p.requires_grad)
                    #     # print(f"After clipping, parameter norm: {param_norm:.2f}")
                    
                    optimizer.step()
                    optimizer.zero_grad()

                    # Check for parameter corruption after optimizer step
                    param_nan_count = 0
                    nan_param_names = []
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any() or torch.isinf(param).any():
                            param_nan_count += 1
                            nan_param_names.append(name)
                            
                    if param_nan_count > 0:
                        print(f"CRITICAL: {param_nan_count} parameters corrupted at epoch {epoch}, batch {b_count}")
                        print(f"Corrupted parameters: {nan_param_names[:5]}...")
                        print("Model parameters are corrupted. Training should be stopped and restarted from checkpoint.")
                        break

                    if rank == 0 and not config.training.debug:
                        # OPTIMIZED: Only sync what's needed for logging
                        step_metrics = {
                            "step": global_step + b_count,
                            "train_step/loss_total": float(loss.item()),
                            "train_step/loss_cat_pos": float(l_cat_pos.item()),
                            "train_step/loss_cat_neg": float(l_cat_neg.item()),
                            "train_step/loss_cat_contrast": float(l_cat_contrast.item()),
                            "train_step/loss_norm_penalty": float((motif_penalty + l2_penalty).item()),
                            "train_step/loss_l2_penalty": float(l2_penalty.item()),
                            "train_step/loss_motif_penalty": float(motif_penalty.item()),
                            "train_step/grad_norm": float(grad_norm),
                            # "train_step/param_norm": param_norm,
                            "train_step/learning_rate": optimizer.param_groups[0]['lr']
                        }

                        if l_str_dist > 0.0:
                            step_metrics.update({
                                "train_step/loss_structure": float(l_str_dist.item()),
                                "train_step/loss_structure_mae": float(key_mae.item()),
                                "train_step/loss_structure_pair": float(l_str_pair.item()),
                                "train_step/loss_keyatm_attmap": float(l_str_attmap.item())
                            })

                        if l_screen > 0.0:
                            step_metrics.update({
                                "train_step/loss_screening": float(l_screen.item()),
                                "train_step/loss_screening_rank": float(l_screen_rank.item()),
                                "train_step/loss_screening_contrast": float(l_screen_cont.item())
                            })

                        wandb.log(step_metrics)

                    # OPTIMIZED: Print with minimal GPU->CPU sync
                    cpu_losses = loss_tracker.get_cpu_losses()
                    print (f"Rank {rank} TRAIN Epoch: [{epoch:2d}/{config.training.max_epoch:2d}], "
                          f"Batch: [{b_count:2d}/{len(loader):2d}], loss: {np.sum(cpu_losses['total'][-accum:]):8.3f} "
                          f"(Category Pos/Neg/Contrast: {np.sum(cpu_losses['CatP'][-accum:]) if len(cpu_losses['CatP']) > 0 else 0:6.1f}/"
                          f"{np.sum(cpu_losses['CatN'][-accum:]) if len(cpu_losses['CatN']) > 0 else 0:6.1f}/"
                          f"{np.sum(cpu_losses['CatCont'][-accum:]) if len(cpu_losses['CatCont']) > 0 else 0:6.3f}|"
                          f"Struct Dist/Mae/Pair: {np.sum(cpu_losses['Str'][-accum:]) if len(cpu_losses['Str']) > 0 else 0:6.1f}/"
                          f"{np.sum(cpu_losses['StrMAE'][-accum:]) if len(cpu_losses['StrMAE']) > 0 else 0:5.2f}/"
                          f"{np.sum(cpu_losses['StrPair'][-accum:]) if len(cpu_losses['StrPair']) > 0 else 0:6.3f}|"
                          f"Screening bce/rank/Contrast: {np.sum(cpu_losses['Screen'][-accum:]) if len(cpu_losses['Screen']) > 0 else 0:4.2f}/"
                          f"{np.sum(cpu_losses['ScreenR'][-accum:]) if len(cpu_losses['ScreenR']) > 0 else 0:4.2f}/"
                          f"{np.sum(cpu_losses['ScreenC'][-accum:]) if len(cpu_losses['ScreenC']) > 0 else 0:6.3f}|"
                          f"{pnames[0]}:", ' '.join(Pbind), ','.join(info['ligands'][0]))

                elif (b_count + 1) % accum == 0:
                    if rank == 0 and not config.training.debug:
                        step_metrics = {
                            "step": global_step + b_count,
                            "valid_step/loss_total": float(loss.item()),
                            "valid_step/loss_cat_pos": float(l_cat_pos.item()),
                            "valid_step/loss_cat_neg": float(l_cat_neg.item()),
                            "valid_step/loss_cat_contrast": float(l_cat_contrast.item()),
                            "valid_step/loss_norm_penalty": float((motif_penalty + l2_penalty).item())
                        }

                        if l_str_dist > 0.0:
                            step_metrics.update({
                                "valid_step/loss_structure": float(l_str_dist.item()),
                                "valid_step/loss_structure_mae": float(key_mae.item()),
                                "valid_step/loss_structure_pair": float(l_str_pair.item()),
                                "valid_step/loss_keyatm_attmap": float(l_str_attmap.item())
                            })

                        if l_screen > 0.0:
                            step_metrics.update({
                                "valid_step/loss_screening": float(l_screen.item()),
                                "valid_step/loss_screening_rank": float(l_screen_rank.item()),
                                "valid_step/loss_screening_contrast": float(l_screen_cont.item())
                            })

                        wandb.log(step_metrics)

                    cpu_losses = loss_tracker.get_cpu_losses()
                    print (f"Rank {rank} VALID Epoch: [{epoch:2d}/{config.training.max_epoch:2d}], "
                          f"Batch: [{b_count:2d}/{len(loader):2d}], loss: {np.sum(cpu_losses['total'][-accum:]):8.3f} "
                          f"(Category Pos/Neg/Contrast: {np.sum(cpu_losses['CatP'][-accum:]) if len(cpu_losses['CatP']) > 0 else 0:6.1f}/"
                          f"{np.sum(cpu_losses['CatN'][-accum:]) if len(cpu_losses['CatN']) > 0 else 0:6.1f}/"
                          f"{np.sum(cpu_losses['CatCont'][-accum:]) if len(cpu_losses['CatCont']) > 0 else 0:6.3f}|"
                          f"Struct Dist/Mae/Pair: {np.sum(cpu_losses['Str'][-accum:]) if len(cpu_losses['Str']) > 0 else 0:6.1f}/"
                          f"{np.sum(cpu_losses['StrMAE'][-accum:]) if len(cpu_losses['StrMAE']) > 0 else 0:5.2f}/"
                          f"{np.sum(cpu_losses['StrPair'][-accum:]) if len(cpu_losses['StrPair']) > 0 else 0:6.3f}|"
                          f"Screening bce/rank/Contrast: {np.sum(cpu_losses['Screen'][-accum:]) if len(cpu_losses['Screen']) > 0 else 0:4.2f}/"
                          f"{np.sum(cpu_losses['ScreenR'][-accum:]) if len(cpu_losses['ScreenR']) > 0 else 0:4.2f}/"
                          f"{np.sum(cpu_losses['ScreenC'][-accum:]) if len(cpu_losses['ScreenC']) > 0 else 0:6.3f}|"
                          f"{pnames[0]}", ' '.join(Pbind), ','.join(info['ligands'][0]))

                b_count += 1

    # OPTIMIZED: Single final CPU sync at epoch end
    final_cpu_losses = loss_tracker.get_cpu_losses()
    return final_cpu_losses, Pt, Pf


def train_model(rank, world_size, config: Config):
    gpu = rank % world_size
    dist.init_process_group(backend='gloo', world_size=world_size, rank=rank)

    if rank == 0 and not config.training.debug:
        wandb.init(
            project="motifscreen-aff",
            name=f"{config.modelname}{config.version}_{config.model_note}",
            mode=config.training.wandb_mode,
            config={
                "model_version": config.version,
                "model_name": config.modelname,
                "learning_rate": config.training.lr,
                "weight_decay": config.training.weight_decay,
                "max_epochs": config.training.max_epoch,
                "batch_size": config.dataloader.batch_size,
                "dropout_rate": config.dropout_rate,
                "edge_mode": config.graph.edgemode,
                "ball_radius": config.graph.ball_radius,
                "w_cat": config.losses.w_cat,
                "w_str": config.losses.w_str,
                "w_screen": config.losses.w_screen
            }
        )

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    model, optimizer, scheduler, start_epoch, train_loss, valid_loss = load_params(rank, config)

    if config.training.ddp:
        if torch.cuda.is_available():
            ddp_model = DDP(model, device_ids=[gpu], find_unused_parameters=False)
        else:
            ddp_model = DDP(model, find_unused_parameters=False)

    if config.training.debug:
        train_datasetf = 'data/small.txt'
        valid_datasetf = 'data/small.txt'
    else:
        train_datasetf = config.train_file
        valid_datasetf = config.valid_file

    train_loader, weights_train = load_data(train_datasetf, world_size, rank, config)
    valid_loader, weights_valid = load_data(valid_datasetf, world_size, rank, config)

    auc_train = {'chembl': [], 'biolip': [], 'pdbbind': []}
    auc_valid = {'chembl': [], 'biolip': [], 'pdbbind': []}

    global_step = start_epoch * len(train_loader)

    for epoch in range(start_epoch, config.training.max_epoch):
        print(f"\n=== Epoch {epoch}/{config.training.max_epoch} ===")

        if config.training.ddp:
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

            if not config.training.debug:
                train_metrics = {
                    "epoch": epoch,
                    "train/loss_total": np.mean(train_loss['total'][-1]),
                    "train/loss_cat_pos": np.mean(train_loss['CatP'][-1]),
                    "train/loss_cat_neg": np.mean(train_loss['CatN'][-1]),
                    "train/loss_cat_contrast": np.mean(train_loss['CatCont'][-1]),
                    "train/loss_norm_penalty": np.mean(train_loss['NormPenalty'][-1]),
                    "train/loss_l2_penalty": np.mean(train_loss['L2Penalty'][-1]),
                    "train/loss_motif_penalty": np.mean(train_loss['MotifPenalty'][-1])
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

                for key in ['pdbbind', 'chembl', 'biolip']:
                    if key in auc_train and len(auc_train[key]) > 0:
                        train_metrics[f"train/auc_{key}"] = auc_train[key][-1]

                wandb.log(train_metrics)

        optimizer.zero_grad()
        global_step += len(train_loader)

        with torch.no_grad():
            if config.training.ddp:
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

            if not config.training.debug:
                valid_metrics = {
                    "epoch": epoch,
                    "valid/loss_total": np.mean(valid_loss['total'][-1]),
                    "valid/loss_cat_pos": np.mean(valid_loss['CatP'][-1]),
                    "valid/loss_cat_neg": np.mean(valid_loss['CatN'][-1]),
                    "valid/loss_cat_contrast": np.mean(valid_loss['CatCont'][-1]),
                    "valid/loss_norm_penalty": np.mean(valid_loss['NormPenalty'][-1]),
                    "valid/loss_l2_penalty": np.mean(valid_loss['L2Penalty'][-1]),
                    "valid/loss_motif_penalty": np.mean(valid_loss['MotifPenalty'][-1])
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

                for key in ['pdbbind', 'chembl', 'biolip']:
                    if key in auc_valid and len(auc_valid[key]) > 0:
                        valid_metrics[f"valid/auc_{key}"] = auc_valid[key][-1]

                wandb.log(valid_metrics)

        print("***SUM***")
        print(f"Train loss | {np.mean(train_loss['total'][-1]):7.4f} | Valid loss | {np.mean(valid_loss['total'][-1]):7.4f}")
        
        # Update learning rate scheduler based on validation loss
        if rank == 0 and scheduler is not None:
            if config.training.scheduler.scheduler_type == "ReduceLROnPlateau":
                scheduler.step(np.mean(valid_loss['total'][-1]))
            elif config.training.scheduler.scheduler_type in ["StepLR", "CosineAnnealingLR"]:
                scheduler.step()

        if rank == 0:
            auc_l = "AUC: "
            for key in ['pdbbind', 'chembl', 'biolip']:
                if key in auc_train and len(auc_train[key]) > 0:
                    auc_l += f' {key} {auc_train[key][-1]:6.4f}'
                if key in auc_valid and len(auc_valid[key]) > 0:
                    auc_l += f' / {auc_valid[key][-1]:6.4f}'
            print(auc_l)

        if rank == 0:
            model_dir = join("models", f"{config.modelname}{config.version}")

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

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'auc_train': auc_train,
                'auc_valid': auc_valid,
            }, join(model_dir, "model.pkl"))

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
    parser = argparse.ArgumentParser(description='Train MotifScreen-Aff (Optimized)')
    parser.add_argument('--config', type=str, default='common',
                        help='Config name (e.g., common) or path to config file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    parser.add_argument('--version', type=str, help='Override model version (e.g., v1.0, v2.0)')
    parser.add_argument('--model_note', type=str, default='',
                        help='Additional note for the model name in wandb')
    args = parser.parse_args()

    # Load configuration
    if args.config.endswith('.yaml'):
        config = load_config(args.config)
    else:
        try:
            config = load_config_with_base(args.config)
        except FileNotFoundError:
            print(f"Config '{args.config}' not found. Using default common config.")
            raise FileNotFoundError(f"Config '{args.config}' not found. Using default common config.")
    if args.debug:
        config.training.debug = True
        config.dataloader.num_workers = 1
    if args.version:
        config.version = args.version
    if args.model_note:
        config.model_note = args.model_note
    print(f"DGL version: {dgl.__version__}")
    print(f"Using config: {args.config}")
    print(f"Using model: MSK{config.version}")
    print(f"Training dropout: {config.dropout_rate}")

    print("\n=== Grouped Parameter Configuration ===")
    print(f"Graph: edgemode={config.graph.edgemode}, edgek={config.graph.edgek}, "
          f"edgedist={config.graph.edgedist}, ball_radius={config.graph.ball_radius}")
    print(f"Processing: ntype={config.processing.ntype}, max_subset={config.processing.max_subset}, "
          f"drop_H={config.processing.drop_H}")
    print(f"Augmentation: randomize={config.augmentation.randomize}")
    print(f"Cross-validation: load_cross={config.cross_validation.load_cross}, "
          f"cross_eval_struct={config.cross_validation.cross_eval_struct}")
    print("="*50)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mp.freeze_support()

    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Using {world_size} GPUs.." if torch.cuda.is_available() else "Using CPU only..")

    if not torch.cuda.is_available():
        config.training.ddp = False
        print("Disabled DDP for CPU-only execution")

    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12346'

    if config.training.ddp:
        mp.spawn(train_model, args=(world_size, config), nprocs=world_size, join=True)
    else:
        train_model(0, 1, config)


if __name__ == "__main__":
    main()
