#!/usr/bin/env python
"""
Complete inference pipeline for MotifScreen-Aff model.

Users provide a config YAML file with:
- protein_pdb: Path to protein PDB file
- center: Either [x, y, z] coordinates or path to crystal ligand file
- ligands_file: Path to batch MOL2 or PDB file containing ligands to screen
- model_path: Path to trained model checkpoint
- output_dir: Directory to save results and intermediate files

The pipeline will:
1. Process protein PDB and generate receptor features (prop.npz) 
2. Generate grid points around binding center (grid.npz)
3. Process ligands to extract key atoms (keyatom.def.npz)
4. Create inference dataset with flexible file paths
5. Load trained model and run inference
6. Output binding predictions and motif predictions
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.io.protein_featurizer import runner as protein_runner
from src.io.ligand_processer import launch_batched_ligand
from src.data.dataset_inference import InferenceDataSet, collate
from src.model.models.msk1 import EndtoEndModel as MSK_1
from src.model.models.msk_v2 import EndtoEndModel as MSK_2
from configs.config_loader import load_config_with_base, Config


class MotifScreenInference:
    """Complete inference pipeline for MotifScreen-Aff"""
    
    def __init__(self, config_file: str):
        """Initialize with inference config file"""
        self.config_file = config_file
        self.inference_config = self._load_inference_config()
        self.model_config = self._load_model_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Create output directory
        self.output_dir = Path(self.inference_config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Inference initialized. Device: {self.device}")
        
    def _load_inference_config(self) -> Dict:
        """Load inference configuration from YAML"""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required_fields = ['protein_pdb', 'ligands_file', 'model_path', 'output_dir']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in config: {field}")
                
        # Validate that either center or crystal_ligand is provided
        if 'center' not in config and 'crystal_ligand' not in config:
            raise ValueError("Either 'center' coordinates or 'crystal_ligand' file must be provided in config")
                
        return config
        
    def _load_model_config(self) -> Config:
        """Load model configuration"""
        config_name = self.inference_config.get('config_name', 'common')
        try:
            return load_config_with_base(config_name)
        except FileNotFoundError:
            print(f"Model config '{config_name}' not found. Using default config.")
            return Config()
            
    def process_protein_and_grid(self) -> tuple[str, str]:
        """Process protein PDB and generate grid using enhanced runner"""
        print("="*60)
        print("Step 1: Processing protein and generating grid...")
        print("="*60)
        
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
            
            # Verify files were created
            if not os.path.exists(prop_file):
                raise FileNotFoundError(f"Receptor properties file not created: {prop_file}")
            if not os.path.exists(grid_file):
                raise FileNotFoundError(f"Grid file not created: {grid_file}")
                
            print(f"✓ Receptor features saved: {prop_file}")
            print(f"✓ Grid points saved: {grid_file}")
            
            return prop_file, grid_file
            
        except Exception as e:
            print(f"✗ Error during protein processing: {e}")
            raise
            
    def process_ligands(self) -> str:
        """Process ligands to extract key atoms"""
        print("="*60)
        print("Step 2: Processing ligands...")
        print("="*60)
        
        ligands_file = self.inference_config['ligands_file']
        if not os.path.exists(ligands_file):
            raise FileNotFoundError(f"Ligands file not found: {ligands_file}")
            
        keyatom_file = self.output_dir / "keyatom.def.npz"
        
        try:
            launch_batched_ligand(ligands_file, N=4, collated_npz=str(keyatom_file))
            
            if not os.path.exists(keyatom_file):
                raise FileNotFoundError(f"Key atoms file not created: {keyatom_file}")
                
            print(f"✓ Key atoms extracted: {keyatom_file}")
            return str(keyatom_file)
            
        except Exception as e:
            print(f"✗ Error during ligand processing: {e}")
            raise
            
    def load_model(self):
        """Load the trained model from checkpoint"""
        print("="*60)
        print("Step 3: Loading trained model...")
        print("="*60)
        
        model_path = self.inference_config['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
        print(f"Loading model from: {model_path}")
        print(f"Model version: {self.model_config.version}")
        
        # Initialize model
        if self.model_config.version == "v1.0":
            self.model = MSK_1(self.model_config)
        elif self.model_config.version == "v2.0":
            self.model = MSK_2(self.model_config)
        else:
            raise ValueError(f"Unsupported model version: {self.model_config.version}")
            
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
        self.model.eval()
        
        print("✓ Model loaded successfully")
        
    def create_dataset(self, prop_file: str, grid_file: str, keyatom_file: str) -> InferenceDataSet:
        """Create inference dataset"""
        print("="*60)
        print("Step 4: Creating inference dataset...")
        print("="*60)
        
        # Get batch size from config or use default
        batch_size = self.inference_config.get('batch_size', self.model_config.processing.max_subset)
        
        dataset = InferenceDataSet(
            protein_pdb=self.inference_config['protein_pdb'],
            prop_npz=prop_file,
            grid_npz=grid_file,
            keyatom_npz=keyatom_file,
            ligands_file=self.inference_config['ligands_file'],
            config=self.model_config,
            batch_size=batch_size
        )
        
        print(f"✓ Dataset created with {dataset.total_ligands} ligands in {dataset.num_batches} batches")
        return dataset
        
    def run_inference(self, dataset: InferenceDataSet) -> Dict:
        """Run model inference on the dataset"""
        print("="*60)
        print("Step 5: Running model inference...")
        print("="*60)
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Create dataloader
        from torch.utils.data import DataLoader
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
                    
                    # Move to device
                    Grec = Grec.to(self.device)
                    Glig = Glig.to(self.device) if Glig is not None else None
                    keyidx = [ki.to(self.device) if torch.is_tensor(ki) else ki for ki in keyidx] if keyidx is not None else None
                    grididx = grididx.to(self.device)
                    
                    batch_ligands = info['ligands'][0] if 'ligands' in info else []
                    print(f"Processing batch {batch_idx}/{len(dataloader)-1} with {len(batch_ligands)} ligands...")
                    print(f"  Receptor nodes: {Grec.number_of_nodes()}")
                    print(f"  Ligand nodes: {Glig.number_of_nodes() if Glig is not None else 0}")
                    print(f"  Grid points: {len(grididx)}")
                    
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
                        print(f"✓ Binding predictions: {len(binding_scores)} ligands")
                        
                    # Process motif predictions (only save from first batch to avoid huge files)
                    if motif_pred is not None and batch_idx == 0:
                        motif_scores = torch.sigmoid(motif_pred).cpu().numpy()
                        results['motif_predictions'] = motif_scores.tolist()
                        print(f"✓ Motif predictions saved: {motif_scores.shape}")
                        
                    # Process coordinate predictions (only save from first batch)
                    if keyxyz_pred is not None and batch_idx == 0:
                        pred_coords = keyxyz_pred.cpu().numpy()
                        results['predicted_coordinates'] = pred_coords.tolist()
                        print(f"✓ Coordinate predictions saved: {pred_coords.shape}")
                        
                    # Collect ligand information
                    results['ligands'].extend(batch_ligands)
                    
                    print(f"✓ Processed batch {batch_idx} successfully")
                    
                except Exception as e:
                    print(f"✗ Error processing batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        return results
        
    def save_results(self, results: Dict):
        """Save inference results to files"""
        print("="*60)
        print("Step 6: Saving results...")
        print("="*60)
        
        # Save complete results as JSON
        results_file = self.output_dir / "inference_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Complete results saved: {results_file}")
        
        # Save binding predictions as CSV
        if results['ligands'] and results['binding_scores']:
            csv_file = self.output_dir / "binding_predictions.csv"
            
            ligands = results['ligands']
            scores = results['binding_scores']
            
            # Create ligand-score pairs and rank them
            ligand_scores = list(zip(ligands, scores))
            ligand_scores.sort(key=lambda x: x[1], reverse=True)
            
            with open(csv_file, 'w') as f:
                f.write("molecule_name,binding_score,rank\n")
                for rank, (ligand, score) in enumerate(ligand_scores, 1):
                    f.write(f"{ligand},{score:.6f},{rank}\n")
                    
            print(f"✓ Binding predictions CSV: {csv_file}")
            
            # Print top 10 predictions
            print("\nTop 10 binding predictions:")
            print("-" * 50)
            print(f"{'Rank':<6} {'Molecule':<20} {'Score':<10}")
            print("-" * 50)
            for rank, (ligand, score) in enumerate(ligand_scores[:10], 1):
                print(f"{rank:<6} {ligand:<20} {score:<10.6f}")
        
        # Save motif predictions if available
        if results['motif_predictions']:
            motif_file = self.output_dir / "motif_predictions.npy"
            np.save(motif_file, np.array(results['motif_predictions']))
            print(f"✓ Motif predictions saved: {motif_file}")
            
        # Save coordinate predictions if available  
        if results['predicted_coordinates']:
            coords_file = self.output_dir / "predicted_coordinates.npy"
            np.save(coords_file, np.array(results['predicted_coordinates']))
            print(f"✓ Coordinate predictions saved: {coords_file}")
            
    def run_complete_pipeline(self):
        """Run the complete inference pipeline"""
        print("MotifScreen-Aff Inference Pipeline")
        print("="*60)
        print(f"Config file: {self.config_file}")
        print(f"Protein PDB: {self.inference_config['protein_pdb']}")
        
        # Print center info based on what's provided
        if 'center' in self.inference_config:
            print(f"Binding center: {self.inference_config['center']}")
        if 'crystal_ligand' in self.inference_config:
            print(f"Crystal ligand: {self.inference_config['crystal_ligand']}")
            
        print(f"Ligands file: {self.inference_config['ligands_file']}")
        print(f"Model path: {self.inference_config['model_path']}")
        print(f"Output directory: {self.output_dir}")
        print("="*60)
        
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
            
            print("="*60)
            print("✓ Inference pipeline completed successfully!")
            print("="*60)
            print(f"Results saved in: {self.output_dir}")
            print(f"- Complete results: {self.output_dir}/inference_results.json")
            print(f"- Binding predictions: {self.output_dir}/binding_predictions.csv")
            if results['motif_predictions']:
                print(f"- Motif predictions: {self.output_dir}/motif_predictions.npy")
            if results['predicted_coordinates']:
                print(f"- Coordinate predictions: {self.output_dir}/predicted_coordinates.npy")
            print("="*60)
            
        except Exception as e:
            print(f"✗ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def main():
    parser = argparse.ArgumentParser(description='MotifScreen-Aff Complete Inference Pipeline')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to inference config YAML file')
    
    args = parser.parse_args()
    
    # Validate config file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Run inference
    inference = MotifScreenInference(args.config)
    inference.run_complete_pipeline()


if __name__ == "__main__":
    main()