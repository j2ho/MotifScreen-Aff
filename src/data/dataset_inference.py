# src/data/dataset_inference.py

import os
import copy
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import dgl
import scipy.spatial

# Local imports
import src.data.types as types
import src.data.utils as myutils
import src.data.kappaidx as kappaidx
from configs.config_loader import Config, DataPathsConfig, GraphParamsConfig, DataProcessingConfig, DataAugmentationConfig

# Reuse components from training dataset
from src.data.dataset import MolecularLoader, GraphBuilder, collate

logger = logging.getLogger(__name__)

class InferenceDataSet(torch.utils.data.Dataset):
    """
    Inference dataset that accepts flexible file paths directly from user.
    
    Unlike TrainingDataSet, this doesn't rely on hardcoded directory structures
    and allows users to specify exact paths to their files.
    
    For inference, all ligands are processed (no random subsampling) and batched
    sequentially using the same receptor graph for each batch.
    """
    
    def __init__(self, 
                 protein_pdb: str,
                 prop_npz: str,
                 grid_npz: str,
                 keyatom_npz: str,
                 ligands_file: str,
                 config: Config = None,
                 batch_size: int = None):
        """
        Initialize inference dataset with direct file paths
        
        Args:
            protein_pdb: Path to protein PDB file
            prop_npz: Path to receptor properties NPZ file  
            grid_npz: Path to grid points NPZ file
            keyatom_npz: Path to key atoms NPZ file
            ligands_file: Path to ligand file (MOL2 or PDB)
            config: Model configuration
            batch_size: Number of ligands per batch (default: use max_subset from config)
        """
        if config is None:
            config = Config()
            
        self.config = config
        self.protein_pdb = protein_pdb
        self.prop_npz = prop_npz
        self.grid_npz = grid_npz
        self.keyatom_npz = keyatom_npz
        self.ligands_file = ligands_file
        self.batch_size = batch_size or self.config.processing.max_subset
        
        # Validate input files
        self._validate_files()
        
        # Initialize components
        self.loader = MolecularLoader(
            config_paths=self.config.paths,
            config_processing=self.config.processing,
            config_augmentation=self.config.augmentation
        )
        self.graph_builder = GraphBuilder(
            config_graph=self.config.graph,
            config_augmentation=self.config.augmentation,
            config_processing=self.config.processing
        )
        
        # Load data
        self.keyatoms_dict = self._load_keyatoms()
        self.grid_data = self._load_grid_data()
        self.ligand_data = self._load_ligand_data()
        
        # Calculate number of batches needed
        self.total_ligands = len(self.ligand_data['tags'])
        self.num_batches = (self.total_ligands + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Inference dataset initialized with {self.total_ligands} ligands")
        logger.info(f"Using batch size {self.batch_size}, total batches: {self.num_batches}")
        
    def _validate_files(self):
        """Validate that all required files exist"""
        files_to_check = {
            'Protein PDB': self.protein_pdb,
            'Receptor properties': self.prop_npz,
            'Grid points': self.grid_npz,
            'Key atoms': self.keyatom_npz,
            'Ligands file': self.ligands_file
        }
        
        for file_type, filepath in files_to_check.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"{file_type} file not found: {filepath}")
                
    def _load_keyatoms(self) -> Dict[str, List[str]]:
        """Load key atoms from NPZ file"""
        try:
            data = np.load(self.keyatom_npz, allow_pickle=True)
            if 'keyatms' in data:
                return data['keyatms'].item()
            return data
        except Exception as e:
            logger.error(f"Failed to load keyatoms from {self.keyatom_npz}: {e}")
            return {}
            
    def _load_grid_data(self) -> Dict:
        """Load grid data from NPZ file"""
        try:
            sample = np.load(self.grid_npz, allow_pickle=True)
            grids = sample['xyz']
            
            cats, mask = None, None
            if 'labels' in sample or 'label' in sample:
                cats = sample.get('labels', sample.get('label'))
                if len(cats) > 0:
                    if cats.shape[1] > self.config.processing.ntype:
                        cats = cats[:, :self.config.processing.ntype]
                    mask = np.sum(cats > 0, axis=1)
                    cats = torch.tensor(cats).float()
                    mask = torch.tensor(mask).float()
                    
            return {
                'grids': grids,
                'cats': cats,
                'mask': mask
            }
        except Exception as e:
            logger.error(f"Error loading grid data from {self.grid_npz}: {e}")
            raise
            
    def _load_ligand_data(self) -> Dict:
        """Load ligand data from file"""
        ligand_file = Path(self.ligands_file)
        
        if ligand_file.suffix.lower() == '.mol2':
            return self._load_mol2_ligands()
        elif ligand_file.suffix.lower() == '.pdb':
            return self._load_pdb_ligands()
        else:
            raise ValueError(f"Unsupported ligand file format: {ligand_file.suffix}")
            
    def _load_mol2_ligands(self) -> Dict:
        """Load ligands from MOL2 file"""
        try:
            # Read all ligands from MOL2 file
            mol_data = myutils.read_mol2_batch(self.ligands_file, drop_H=self.config.processing.drop_H)
            
            if mol_data is None:
                raise ValueError(f"Failed to read MOL2 file: {self.ligands_file}")
                
            elems, qs, bonds, borders, xyz, nneighs, atms, atypes, tags_read = mol_data
            
            return {
                'elems': elems,
                'qs': qs, 
                'bonds': bonds,
                'borders': borders,
                'xyz': xyz,
                'nneighs': nneighs,
                'atms': atms,
                'atypes': atypes,
                'tags': tags_read,
                'type': 'batch'
            }
            
        except Exception as e:
            logger.error(f"Error loading MOL2 ligands from {self.ligands_file}: {e}")
            raise
            
    def _load_pdb_ligands(self) -> Dict:
        """Load ligands from PDB file (assuming single ligand)"""
        try:
            mol_data = self.loader.read_mol2_single(self.ligands_file)  # This also works for PDB
            if mol_data is None:
                raise ValueError(f"Failed to read PDB file: {self.ligands_file}")
                
            # Convert single ligand to batch format for consistency
            ligand_name = Path(self.ligands_file).stem
            
            return {
                'elems': [mol_data[0]],
                'qs': [mol_data[1]],
                'bonds': [mol_data[2]], 
                'borders': [mol_data[3]],
                'xyz': [mol_data[4]],
                'nneighs': [mol_data[5]],
                'atms': [mol_data[6]],
                'atypes': [mol_data[7]],
                'tags': [ligand_name],
                'type': 'single'
            }
            
        except Exception as e:
            logger.error(f"Error loading PDB ligand from {self.ligands_file}: {e}")
            raise
            
    def __len__(self) -> int:
        """Return number of batches"""
        return self.num_batches
        
    def __getitem__(self, batch_index: int) -> Optional[Tuple]:
        """Get inference batch"""
        if batch_index >= self.num_batches:
            return None
            
        try:
            return self._get_inference_batch(batch_index)
        except Exception as e:
            logger.error(f"Error processing inference batch {batch_index}: {e}")
            traceback.print_exc()
            return None
            
    def _get_inference_batch(self, batch_index: int) -> Tuple:
        """Process and return inference batch"""
        
        # Calculate ligand indices for this batch
        start_idx = batch_index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.total_ligands)
        ligand_indices = list(range(start_idx, end_idx))
        
        # Process ligands into graphs for this batch
        ligand_graphs, key_indices_list, binding_labels, batch_tags = self._process_ligand_batch(ligand_indices)
        
        if not ligand_graphs:
            logger.error(f"No valid ligand graphs generated for batch {batch_index}")
            return self._get_null_result()
            
        # Calculate origin from grid center (same for all batches)
        grids = self.grid_data['grids']
        origin = torch.tensor(np.mean(grids, axis=0)).float()
        
        # Center ligand graphs around origin
        for graph in ligand_graphs:
            if graph.ndata['x'].dim() == 3:  # (N, 1, 3)
                graph.ndata['x'] = graph.ndata['x'] - origin.view(1, 1, 3)
            else:  # (N, 3)
                graph.ndata['x'] = graph.ndata['x'] - origin.view(1, 3)
        
        # Center grids around origin
        grids_centered = grids - origin.numpy()
        
        # Build receptor graph (same for all batches, but we rebuild each time for consistency)
        receptor_graph, processed_grids, grid_indices = self.graph_builder.build_receptor_graph(
            self.prop_npz, grids_centered, origin, gridchain=None
        )
        
        if receptor_graph is None:
            logger.error("Failed to build receptor graph")
            return self._get_null_result()
            
        # Build info dictionary
        info = {
            'pname': Path(self.protein_pdb).stem,
            'name': f"inference/{Path(self.protein_pdb).stem}",
            'com': origin,
            'grididx': grid_indices,
            'grid': processed_grids,
            'gridinfo': self.grid_npz,
            'source': 'inference',
            'eval_struct': 1,
            'ligands': batch_tags,
            'nK': [len(idx) for idx in key_indices_list],
            'atms': [self.ligand_data['atms'][i] for i in ligand_indices]
        }
        
        # Get key coordinates (use first ligand as reference)
        key_xyz = torch.zeros((4, 3))
        if key_indices_list and len(key_indices_list[0]) > 0:
            first_graph = ligand_graphs[0]
            first_key_indices = key_indices_list[0]
            if hasattr(first_graph, 'ndata') and 'x' in first_graph.ndata:
                coords = first_graph.ndata['x']
                if coords.dim() == 3:  # (N, 1, 3)
                    key_xyz = coords[first_key_indices].squeeze(1)  # (K, 3)
                else:  # (N, 3)
                    key_xyz = coords[first_key_indices]  # (K, 3)
                    
                # Pad or truncate to 4 coordinates
                if len(key_xyz) > 4:
                    key_xyz = key_xyz[:4]
                elif len(key_xyz) < 4:
                    padding = torch.zeros((4 - len(key_xyz), 3))
                    key_xyz = torch.cat([key_xyz, padding], dim=0)
        
        return (
            receptor_graph,           # Receptor graph
            ligand_graphs,           # List of ligand graphs
            self.grid_data['cats'],  # Category labels (optional)
            self.grid_data['mask'],  # Mask for categories (optional)
            key_xyz,                 # Key atom coordinates
            key_indices_list,        # Key atom indices for each ligand
            binding_labels,          # Binding labels (dummy for inference)
            info                     # Information dictionary
        )
        
    def _process_ligand_batch(self, ligand_indices: List[int]) -> Tuple[List, List, List, List]:
        """Process a batch of ligands into graphs and extract key atoms"""
        ligand_graphs = []
        key_indices_list = []
        binding_labels = []
        batch_tags = []
        
        ligand_data = self.ligand_data
        
        for i in ligand_indices:
            tag = ligand_data['tags'][i]
            try:
                # Build molecular graph
                mol_tuple = (
                    ligand_data['elems'][i],
                    ligand_data['qs'][i], 
                    ligand_data['bonds'][i],
                    ligand_data['borders'][i],
                    ligand_data['xyz'][i],
                    ligand_data['nneighs'][i],
                    ligand_data['atypes'][i]
                )
                
                graph = self.graph_builder.build_ligand_graph(mol_tuple, name=tag)
                if graph is None:
                    logger.warning(f"Failed to build graph for ligand: {tag}")
                    continue
                    
                # Center graph coordinates
                com = torch.mean(graph.ndata['x'], axis=0).float()
                graph.ndata['x'] = (graph.ndata['x'] - com).float()
                
                # Get atom names (filter hydrogens if needed)
                atoms = ligand_data['atms'][i]
                if self.config.processing.drop_H:
                    filtered_atoms = [
                        atom for atom, element in zip(atoms, ligand_data['elems'][i]) 
                        if element != 'H'
                    ]
                else:
                    filtered_atoms = atoms
                    
                # Identify key atoms
                key_indices = self._identify_key_atoms(tag, filtered_atoms)
                if not key_indices:
                    logger.warning(f"No key atoms found for ligand: {tag}")
                    # Use first few atoms as fallback
                    key_indices = list(range(min(4, len(filtered_atoms))))
                    
                ligand_graphs.append(graph)
                key_indices_list.append(key_indices)
                binding_labels.append(0)  # Dummy label for inference
                batch_tags.append(tag)
                
            except Exception as e:
                logger.error(f"Error processing ligand {tag}: {e}")
                continue
                
        return ligand_graphs, key_indices_list, binding_labels, batch_tags
        
    def _identify_key_atoms(self, target: str, atoms: List[str]) -> List[int]:
        """Identify key atoms for a ligand"""
        if target not in self.keyatoms_dict:
            return []
            
        key_indices = [
            atoms.index(atom) for atom in self.keyatoms_dict[target] 
            if atom in atoms
        ]
        
        # Limit to 10 key atoms
        if len(key_indices) > 10:
            key_indices = list(np.random.choice(key_indices, 10, replace=False))
            
        return key_indices
        
    def _get_null_result(self) -> Tuple:
        """Return null result for failed processing"""
        info = {'pname': Path(self.protein_pdb).stem}
        return (None, None, None, None, None, None, None, info)


# Reuse the collate function from training dataset since it handles the same data structure
# The collate function is imported from src.data.dataset