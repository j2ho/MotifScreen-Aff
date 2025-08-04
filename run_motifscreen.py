#!/usr/bin/env python

import os
import sys
import argparse
import numpy as np
import torch
import yaml
import json
import time
from pathlib import Path
from typing import Dict

# Distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.io.protein_featurizer import runner as protein_runner
from src.io.ligand_processer import launch_batched_ligand
from src.data.dataset_inference import InferenceDataSet, collate
from src.model.models.msk1 import EndtoEndModel as MSK_1
from src.model.models.msk_v2 import EndtoEndModel as MSK_2
from configs.config_loader import load_config_with_base, Config
from scripts.train.utils import to_cuda

import logging

class CustomFormatter(logging.Formatter):
    def format(self, record):
        if record.levelno >= logging.ERROR:
            return f"[ERROR] {record.getMessage()}"
        else:
            return f" {record.getMessage()}"

class InfoHandler(logging.StreamHandler):
    """Handler that sends INFO messages to stdout"""
    def __init__(self):
        super().__init__(sys.stdout)
        
    def emit(self, record):
        if record.levelno < logging.ERROR:
            super().emit(record)

class ErrorHandler(logging.StreamHandler):
    """Handler that sends ERROR+ messages to stderr"""
    def __init__(self):
        super().__init__(sys.stderr)
        
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            super().emit(record)

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    # Create separate handlers for INFO (stdout) and ERROR (stderr)
    info_handler = InfoHandler()
    error_handler = ErrorHandler()
    
    formatter = CustomFormatter()
    info_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)
logger.setLevel(logging.INFO)


class MotifScreenInference:
    """Complete inference pipeline for MotifScreen-Aff"""
    
    def __init__(self, config_file: str, rank: int = 0, world_size: int = 1):
        """Initialize with inference config file"""
        self.config_file = config_file
        self.rank = rank
        self.world_size = world_size
        self.inference_config = self._load_inference_config()
        self.model_config = self._load_model_config()
        
        # Multi-GPU setup for distributed inference
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(self.device)
            self.num_gpus = torch.cuda.device_count()
            if rank == 0:  # Only log from main process
                logger.info(f"Found {self.num_gpus} GPU(s)")
                for i in range(self.num_gpus):
                    logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                if world_size > 1:
                    logger.info(f"Using distributed inference with {world_size} processes")
        else:
            self.device = torch.device("cpu")
            self.num_gpus = 0
            if rank == 0:
                logger.info("Using CPU")
            
        self.model = None
        
        # Create output directory (only on main process)
        if rank == 0:
            self.output_dir = Path(self.inference_config['output_dir'])
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Using output directory: {self.output_dir}")
        else:
            self.output_dir = Path(self.inference_config['output_dir'])
            
        self.save_aux = self.inference_config.get('save_aux', False)
        
    def _load_inference_config(self) -> Dict:
        """Load inference configuration from YAML"""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required_fields = ['protein_pdb', 'ligands_file', 'model_path', 'output_dir']
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field in config: {field}")
                raise ValueError
                
        # Validate that either center or crystal_ligand is provided
        if 'center' not in config and 'crystal_ligand' not in config:
            logger.error("Either 'center' coordinates or 'crystal_ligand' file must be provided in config")
            raise ValueError

        return config
        
    def _load_model_config(self) -> Config:
        """Load model configuration"""
        config_name = self.inference_config.get('config_name', 'common')
        try:
            return load_config_with_base(config_name)
        except FileNotFoundError:
            logger.error(f"Model config '{config_name}' not found. Using default config.")
            return Config()
            
    def process_protein_and_grid(self) -> tuple[str, str]:
        logger.info("="*60)
        logger.info("Step 1: Processing protein and generating grid...")
        logger.info("="*60)

        # Prepare config for protein_featurizer runner
        runner_config = {
            'protein_pdb': self.inference_config['protein_pdb'],
            'gridsize': self.inference_config.get('gridsize', 1.5),
            'padding': self.inference_config.get('padding', 10.0),
            'clash': self.inference_config.get('clash', 1.1),
            'output_dir': str(self.output_dir)
        }
        
        # Add center coordinates if provided
        if 'center' in self.inference_config:
            runner_config['center'] = self.inference_config['center']
            
        # Add crystal ligand if provided  
        if 'crystal_ligand' in self.inference_config:
            runner_config['crystal_ligand'] = self.inference_config['crystal_ligand']
        
        try:
            output_prefix = protein_runner(runner_config)
            prop_file = f"{output_prefix}.prop.npz"
            grid_file = f"{output_prefix}.grid.npz"
            
            if not os.path.exists(prop_file):
                raise FileNotFoundError(f"Receptor properties file not created: {prop_file}")
            if not os.path.exists(grid_file):
                raise FileNotFoundError(f"Grid file not created: {grid_file}")

            logger.info(f"Receptor properties saved to: {prop_file}")
            logger.info(f"Grid points saved to: {grid_file}")

            return prop_file, grid_file
            
        except Exception as e:
            logger.error(f"Failed to calculate features for protein: {e}")
            raise
            
    def process_ligands(self) -> str:
        """Process ligands to extract key atoms"""
        logger.info("="*60)
        logger.info("Step 2: Processing ligands...")
        logger.info("="*60)

        ligands_file = self.inference_config['ligands_file']
        if not os.path.exists(ligands_file):
            logger.error(f"Ligands file not found: {ligands_file}")
            raise FileNotFoundError
            
        keyatom_file = self.output_dir / "keyatom.def.npz"
        
        try:
            launch_batched_ligand(ligands_file, N=4, collated_npz=str(keyatom_file))
            
            if not os.path.exists(keyatom_file):
                logger.error(f"Key atoms file not created: {keyatom_file}")
                raise FileNotFoundError

            logger.info(f"Key atoms saved to: {keyatom_file}")
            return str(keyatom_file)
            
        except Exception as e:
            logger.error(f"Error during ligand processing: {e}")
            raise
            
    def load_model(self):
        """Load the trained model from checkpoint"""
        logger.info("="*60)
        logger.info("Step 3: Loading trained model...")
        logger.info("="*60)

        model_path = self.inference_config['model_path']
        if not os.path.exists(model_path):
            logger.error(f"Model checkpoint not found: {model_path}")
            raise FileNotFoundError

        logger.info(f"Loading model from: {model_path}")
        logger.info(f"Model version: {self.model_config.version}")

        # Initialize model
        if self.model_config.version == "v1.0":
            self.model = MSK_1(self.model_config)
        elif self.model_config.version == "v2.0":
            self.model = MSK_2(self.model_config)
        else:
            logger.error(f"Unsupported model version: {self.model_config.version}")
            raise ValueError

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle potential module prefix from DDP training
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if any(key.startswith("module.") for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace("module.", "")
                new_state_dict[new_key] = value
            state_dict = new_state_dict
            
        # Load state dict
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        
        # Enable DDP for multi-GPU inference
        if self.world_size > 1:
            if self.rank == 0:
                logger.info(f"Wrapping model with DistributedDataParallel for {self.world_size} processes")
            self.model = DDP(self.model, device_ids=[self.rank])
        
        self.model.eval()
        if self.rank == 0:
            logger.info("Model loaded successfully")
        
    def create_dataset(self, prop_file: str, grid_file: str, keyatom_file: str) -> InferenceDataSet:
        """Create inference dataset"""
        logger.info("="*60)
        logger.info("Step 4: Creating inference dataset...")
        logger.info("="*60)

        # Get batch size from config or use default, scale with number of GPUs
        base_batch_size = self.inference_config.get('batch_size', self.model_config.processing.max_subset)
        # Scale batch size by number of GPUs for better utilization
        effective_batch_size = base_batch_size * max(1, self.num_gpus)
        
        dataset = InferenceDataSet(
            protein_pdb=self.inference_config['protein_pdb'],
            prop_npz=prop_file,
            grid_npz=grid_file,
            keyatom_npz=keyatom_file,
            ligands_file=self.inference_config['ligands_file'],
            config=self.model_config,
            batch_size=effective_batch_size
        )

        if self.num_gpus > 1:
            logger.info(f"Dataset created with {dataset.total_ligands} ligands in {dataset.num_batches} batches")
            logger.info(f"  Base batch size: {base_batch_size}, Effective batch size: {effective_batch_size} (scaled for {self.num_gpus} GPUs)")
        else:
            logger.info(f"Dataset created with {dataset.total_ligands} ligands in {dataset.num_batches} batches")
        return dataset
        
    def run_inference(self, dataset: InferenceDataSet) -> Dict:
        """Run model inference on the dataset"""
        logger.info("="*60)
        logger.info("Step 5: Running model inference...")
        logger.info("="*60)

        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            raise RuntimeError
            
        # Create dataloader with distributed sampling if using multiple processes
        from torch.utils.data import DataLoader, DistributedSampler
        
        if self.world_size > 1:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, 
                                  collate_fn=collate, num_workers=0)
        else:
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, 
                                  collate_fn=collate, num_workers=0)
        
        results = {
            'ligands': [],
            'binding_scores': [],
            'motif_predictions': [],
            'predicted_coordinates': [],
            'model_info': {
                'version': self.model_config.version,
                'model_path': self.inference_config['model_path'],
                'total_batches': len(dataloader),
                'batch_size': dataset.batch_size
            }
        }
        
        with torch.no_grad():
            for batch_idx, inputs in enumerate(dataloader):
                if inputs is None:
                    print(f"✗ Skipping batch {batch_idx} (None input)")
                    continue
                    
                try:
                    (Grec, Glig, cats, masks, keyxyz, keyidx, blabel, info) = inputs
                    grididx = info['grididx']
                    
                    if any(x is None for x in (Grec, Glig, keyidx, grididx)):
                        print(f"✗ Skipping batch {batch_idx} (missing components)")
                        continue
                    
                    # Move to device using to_cuda utility for proper DGL graph handling
                    Grec = to_cuda(Grec, self.device)
                    Glig = to_cuda(Glig, self.device) if Glig is not None else None
                    keyidx = to_cuda(keyidx, self.device) if keyidx is not None else None
                    grididx = grididx.to(self.device)
                    
                    batch_ligands = info['ligands'][0] if 'ligands' in info else []
                    logger.info(f"Processing batch {batch_idx}/{len(dataloader)-1} with {len(batch_ligands)} ligands...")  
                    
                    # Log GPU memory usage for multi-GPU setups
                    if self.num_gpus > 1 and batch_idx % 10 == 0:  # Every 10 batches
                        memory_info = []
                        for i in range(self.num_gpus):
                            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
                            cached = torch.cuda.memory_reserved(i) / 1024**3     # GB
                            memory_info.append(f"GPU{i}: {allocated:.1f}GB/{cached:.1f}GB")
                        logger.info(f"  Memory usage: {', '.join(memory_info)}")
                    
                    # logger.info(f"  Receptor nodes: {Grec.number_of_nodes()}")
                    # logger.info(f"  Ligand nodes: {Glig.number_of_nodes() if Glig is not None else 0}")
                    # logger.info(f"  Grid points: {len(grididx)}")

                    # Run inference
                    keyxyz_pred, key_pairdist_pred, rec_key_z, motif_pred, bind_pred, absaff_pred = self.model(
                        Grec, Glig, keyidx, grididx,
                        gradient_checkpoint=False,
                        drop_out=False
                    )
                    
                    # Process binding predictions
                    if bind_pred is not None and len(bind_pred) > 0:
                        binding_scores = torch.sigmoid(bind_pred[0]).cpu().numpy()
                        results['binding_scores'].extend(binding_scores.tolist())
                        logger.info(f"Binding predictions: {len(binding_scores)} ligands")

                    # Process motif predictions 
                    if self.save_aux and motif_pred is not None:
                        motif_scores = torch.sigmoid(motif_pred).cpu().numpy()
                        results['motif_predictions'] = motif_scores.tolist()
                        logger.info(f"Motif predictions saved: {motif_scores.shape}")

                    # Process coordinate predictions 
                    if self.save_aux and keyxyz_pred is not None:
                        pred_coords = keyxyz_pred.cpu().numpy()
                        results['predicted_coordinates'] = pred_coords.tolist()
                        logger.info(f"Coordinate predictions saved: {pred_coords.shape}")

                    # Collect ligand information
                    results['ligands'].extend(batch_ligands)

                    logger.info(f"Processed batch {batch_idx} successfully")
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Gather results from all processes if using distributed inference
        if self.world_size > 1:
            # Collect results from all processes
            all_ligands = [None] * self.world_size
            all_binding_scores = [None] * self.world_size
            
            dist.all_gather_object(all_ligands, results['ligands'])
            dist.all_gather_object(all_binding_scores, results['binding_scores'])
            
            if self.rank == 0:
                # Combine results from all processes
                combined_ligands = []
                combined_scores = []
                for ligands, scores in zip(all_ligands, all_binding_scores):
                    combined_ligands.extend(ligands)
                    combined_scores.extend(scores)
                
                results['ligands'] = combined_ligands
                results['binding_scores'] = combined_scores
                
                logger.info(f"Gathered results from {self.world_size} processes")
                logger.info(f"Total ligands processed: {len(combined_ligands)}")
        
        return results
        
    def save_results(self, results: Dict):
        """Save inference results to files"""
        logger.info("="*60)
        logger.info("Step 6: Saving results...")
        logger.info("="*60)

        # Save complete results as JSON
        results_file = self.output_dir / "inference_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Complete results saved: {results_file}")
        
        # Save binding predictions as CSV
        if results['ligands'] and results['binding_scores']:
            csv_file = self.output_dir / "binding_predictions.csv"
            
            ligands = results['ligands']
            scores = results['binding_scores']
            
            # Create ligand-score pairs and rank them
            ligand_scores = list(zip(ligands, scores))
            ligand_scores.sort(key=lambda x: x[1], reverse=True)
            
            with open(csv_file, 'w') as f:
                f.write("rank,compound,binding_score\n")
                for rank, (ligand, score) in enumerate(ligand_scores, 1):
                    f.write(f"{rank},{ligand},{score:.6f}\n")

            logger.info(f"Binding predictions CSV: {csv_file}")

            # Print top 10 predictions
            logger.info("\nTop 10 binding predictions:")
            logger.info("-" * 50)
            logger.info(f"{'Rank':<6} {'Compound':<20} {'Score':<10}")
            logger.info("-" * 50)
            for rank, (ligand, score) in enumerate(ligand_scores[:10], 1):
                logger.info(f"{rank:<6} {ligand:<20} {score:<10.6f}")
        
        if self.save_aux:
            # Save motif predictions if available
            if results['motif_predictions']:
                motif_file = self.output_dir / "motif_predictions.npy"
                np.save(motif_file, np.array(results['motif_predictions']))
                logger.info(f"Motif predictions saved: {motif_file}")
            
            # # Save coordinate predictions if available  
            if results['predicted_coordinates']:
                coords_file = self.output_dir / "predicted_coordinates.npy"
                np.save(coords_file, np.array(results['predicted_coordinates']))
                logger.info(f"Coordinate predictions saved: {coords_file}")
            
    def run_complete_pipeline(self):
        """Run the complete inference pipeline"""
        start_time = time.time()
        logger.info("MotifScreen-Aff Inference Pipeline")
        logger.info("="*60)
        logger.info(f"Config file: {self.config_file}")
        logger.info(f"Protein PDB: {self.inference_config['protein_pdb']}")

        # Print center info based on what's provided
        if 'center' in self.inference_config:
            logger.info(f"Binding center: {self.inference_config['center']}")
        if 'crystal_ligand' in self.inference_config:
            logger.info(f"Crystal ligand: {self.inference_config['crystal_ligand']}")

        logger.info(f"Ligands file: {self.inference_config['ligands_file']}")
        logger.info(f"Model path: {self.inference_config['model_path']}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info("="*60)
        
        try:
            # Step 1 & 2: Process protein and ligands
            prop_file, grid_file = self.process_protein_and_grid()
            keyatom_file = self.process_ligands()
            
            # Step 3: Load model
            self.load_model()
            
            # Step 4: Create dataset
            dataset = self.create_dataset(prop_file, grid_file, keyatom_file)
            
            # Step 5: Run inference
            results = self.run_inference(dataset)
            
            # Step 6: Save results
            self.save_results(results)
            
            # Calculate and display runtime
            end_time = time.time()
            total_runtime = end_time - start_time
            logger.info("="*60)
            logger.info("Inference pipeline completed successfully!")
            logger.info("="*60)
            logger.info(f"Results saved in: {self.output_dir}")
            logger.info(f"- Complete results: {self.output_dir}/inference_results.json")
            logger.info(f"- Binding predictions: {self.output_dir}/binding_predictions.csv")
            if self.save_aux and results['motif_predictions']:
                logger.info(f"- Motif predictions: {self.output_dir}/motif_predictions.npy")
            if self.save_aux and results['predicted_coordinates']:
                logger.info(f"- Coordinate predictions: {self.output_dir}/predicted_coordinates.npy")
            logger.info("="*60)
            logger.info(f"Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.1f} minutes)")
            logger.info("="*60)
            
        except Exception as e:
            end_time = time.time()
            total_runtime = end_time - start_time
            logger.error(f"Pipeline failed after {total_runtime:.2f} seconds: {e}")
            import traceback
            traceback.print_exc()
            raise


def run_inference_worker(rank: int, world_size: int, config_file: str):
    """Worker function for distributed inference"""
    # Initialize distributed process group
    if world_size > 1:
        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12000')
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    
    # Run inference
    inference = MotifScreenInference(config_file, rank, world_size)
    inference.run_complete_pipeline()
    
    # Clean up distributed process group
    if world_size > 1:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser(description='MotifScreen-Aff Complete Inference Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to inference config YAML file')
    
    args = parser.parse_args()
    
    # Validate config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Check for multi-GPU setup
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if world_size > 1:
        # Use multiprocessing spawn for distributed inference
        import torch.multiprocessing as mp
        mp.spawn(run_inference_worker, args=(world_size, args.config), nprocs=world_size, join=True)
    else:
        # Single GPU/CPU inference
        run_inference_worker(0, 1, args.config)


if __name__ == "__main__":
    main()