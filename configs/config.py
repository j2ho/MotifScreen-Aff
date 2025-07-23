"""
Configuration loader for MotifScreen-Aff training
Replaces the args.py system with YAML-based configuration using dataclasses
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
from pathlib import Path


@dataclass
class GridParams:
    """Grid network parameters - corresponds to GridArgs.params"""
    dropout_rate: float = 0.1
    num_layers_grid: int = 2
    l0_in_features: int = 102
    n_heads: int = 4
    num_channels: int = 32
    num_edge_features: int = 3
    l0_out_features: int = 32
    ntypes: int = 6


@dataclass
class LigandParams:
    """Ligand network parameters - corresponds to LigandArgs.params"""
    model: str = "gat"  # Model type for ligand processing
    dropout_rate: float = 0.1
    num_layers: int = 2
    n_heads: int = 4
    num_channels: int = 32
    l0_in_features: int = 15
    l0_out_features: int = 32
    num_edge_features: int = 5
    n_lig_global_in: int = 19  # Number of global ligand input features
    n_lig_global_out: int = 4  # Number of global ligand embeddings


@dataclass
class TRParams:
    dropout_rate: float = 0.1
    num_channels: int = 32
    num_degrees: int = 3
    div: int = 4
    l0_in_features_lig: int = 15
    l0_in_features_rec: int = 32
    l1_in_features: int = 0
    l1_out_features: int = 0
    num_edge_features: int = 5
    l0_out_features_lig: int = 64
    d: int = 64
    m: int = 64
    c: int = 64
    n_trigon_lig_layers: int = 3
    n_trigon_key_layers: int = 3
    shared_trigon: bool = False
    normalize_Xform: bool = True
    lig_to_key_attn: bool = True

@dataclass
class AffModuleParams:
    """Parameters for the Affinity Module"""
    classification_mode: str = "former_contrast"  # Mode for classification

@dataclass
class TrainingConfig:
    """Training configuration"""
    lr: float = 1.0e-4
    max_epoch: int = 500
    debug: bool = False
    ddp: bool = True
    silent: bool = False
    accumulation_steps: int = 1


@dataclass
class DataConfig:
    """Data configuration"""
    datapath: str = "/ml/motifnet/features_com2/"
    train_file: str = "data/PLmix.60k.screen.txt"
    valid_file: str = "data/PLmix.60k.screen.txt"
    affinityf: Optional[str] = None
    decoyf: str = "decoys.BL2.npz"  
    keyatomf: str = "keyatom.def.npz" 
    ball_radius: float = 8.0
    edgedist: List[float] = field(default_factory=lambda: [2.2, 4.5])
    edgemode: str = "topk"
    edgek: List[int] = field(default_factory=lambda: [8, 16])
    randomize: float = 0.2
    ntype: int = 6
    maxedge: int = 40000
    maxnode: int = 3000
    drop_H: bool = True
    max_subset: int = 5


@dataclass
class DataLoaderConfig:
    """Data loader configuration"""
    batch_size: int = 1
    num_workers: int = 5
    pin_memory: bool = True
    shuffle: bool = True


@dataclass
class Config:
    """Main configuration class - corresponds to Argument class"""
    # Model name
    modelname: str = "MSK"
    version: str = "v1.0"
    
    # Parameter groups (using dataclasses instead of dicts)
    params_grid: GridParams = field(default_factory=GridParams)
    params_ligand: LigandParams = field(default_factory=LigandParams)
    params_TR: TRParams = field(default_factory=TRParams)
    params_Aff: AffModuleParams = field(default_factory=AffModuleParams)

    # Main model parameters
    dropout_rate: float = 0.2
    LR: float = 1.0e-4

    struct_loss: str = "mse"
    w_str: float = 0.2
    w_cat: float = 1.0
    w_penalty: float = 1.0e-10
    screenloss: str = "BCE"
    w_contrast: float = 2.0
    w_spread: float = 5.0
    w_screen: float = 0.0
    w_screen_contrast: float = 0.0
    w_screen_ranking: float = 0.0
    w_Dkey: float = 1.0

    pert: bool = False
    load_cross: bool = False
    cross_eval_struct: bool = False
    cross_grid: float = 0.0
    nonnative_struct_weight: float = 0.2
    randomize_grid: float = 0.0
    firstshell_as_grid: bool = False
    use_input_PHcore: bool = False
    
    # Training options
    max_epoch: int = 500
    debug: bool = False
    accumulation_steps: int = 1
    ddp: bool = True
    silent: bool = False
    load_checkpoint: bool = False

    # Data and loader configs
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    def __post_init__(self):
        """Post-initialization to handle parameter adjustments"""
        self._sync_dropout_rates()
        self._apply_ddp_adjustments()

    def _sync_dropout_rates(self):
        """Sync dropout rates across all parameter groups"""
        self.params_grid.dropout_rate = self.dropout_rate
        self.params_ligand.dropout_rate = self.dropout_rate
        self.params_TR.dropout_rate = self.dropout_rate

    def _apply_ddp_adjustments(self):
        """Apply DDP-specific adjustments"""
        if self.ddp:
            self.dataloader.shuffle = False
        
        if self.debug:
            self.dataloader.num_workers = 1

    def set_dropout_rate(self, value: float):
        """Set dropout rate across all parameter groups"""
        self.dropout_rate = value
        self._sync_dropout_rates()


def deep_merge(base_dict: Dict, override_dict: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(config_path: str, base_config_path: Optional[str] = None) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to the main config file
        base_config_path: Path to base config file (optional)
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # If base config is specified, merge it
    if base_config_path:
        base_path = Path(base_config_path)
        if base_path.exists():
            with open(base_path, 'r') as f:
                base_dict = yaml.safe_load(f)
            config_dict = deep_merge(base_dict, config_dict)
    
    # Extract nested parameter configurations
    grid_params = GridParams(**config_dict.get('model', {}).get('grid', {}))
    ligand_params = LigandParams(**config_dict.get('model', {}).get('ligand', {}))
    tr_params = TRParams(**config_dict.get('model', {}).get('TR', {}))
    aff_params = AffModuleParams(**config_dict.get('model', {}).get('aff', {}))

    data_config = DataConfig(**config_dict.get('data', {}))
    dataloader_config = DataLoaderConfig(**config_dict.get('dataloader', {}))
    data_misc_config = config_dict.get('misc', {})
    cv_config = config_dict.get('cross_validation', {})
    
    # Build main config
    main_config = config_dict.get('model', {})
    training_config = config_dict.get('training', {})
    loss_config = config_dict.get('losses', {})
    
    # Create config with all parameters
    config = Config(
        # Model name
        version=main_config.get('version', 'v1.0'),
        modelname=main_config.get('name', 'MSK'),
        
        # Parameter groups
        params_grid=grid_params,
        params_ligand=ligand_params,
        params_TR=tr_params,
        params_Aff=aff_params,
        
        # Main model parameters
        dropout_rate=training_config.get('dropout_rate', 0.2),

        struct_loss=loss_config.get('struct_loss', 'mse'),
        w_str=loss_config.get('w_str', 0.2),
        w_cat=loss_config.get('w_cat', 1.0),
        w_penalty=loss_config.get('w_penalty', 1.0e-10),
        screenloss=loss_config.get('screenloss', 'BCE'),
        w_contrast=loss_config.get('w_contrast', 2.0),
        w_spread=loss_config.get('w_spread', 5.0),
        w_screen=loss_config.get('w_screen', 0.0),
        w_screen_contrast=loss_config.get('w_screen_contrast', 0.0),
        w_screen_ranking=loss_config.get('w_screen_ranking', 0.0),
        w_Dkey=loss_config.get('w_Dkey', 1.0),

        # Training 
        LR=training_config.get('lr', 1.0e-4),
        max_epoch=training_config.get('max_epoch', 500),
        debug=training_config.get('debug', False),
        accumulation_steps=training_config.get('accumulation_steps', 1),

        # Data and loader configs
        data=data_config,
        dataloader=dataloader_config,
        # Dataset configurations other than data/dataloader
        load_cross=cv_config.get('load_cross', False),
        cross_eval_struct=cv_config.get('cross_eval_struct', False),
        cross_grid=cv_config.get('cross_grid', 0.0),
        nonnative_struct_weight=cv_config.get('nonnative_struct_weight', 0.2),
        randomize_grid=data_misc_config.get('randomize_grid', 0.0),
        pert=data_misc_config.get('pert', False),
        firstshell_as_grid=data_misc_config.get('firstshell_as_grid', False),
        use_input_PHcore=data_misc_config.get('use_input_PHcore', False),
        
        # Training options
        ddp=training_config.get('ddp', True),
        silent=training_config.get('silent', False),
        load_checkpoint=training_config.get('load_checkpoint', False)
    )
    
    # Handle datasetf as list for train/valid split
    data_config_dict = config_dict.get('data', {})
    if 'train_file' in data_config_dict and 'valid_file' in data_config_dict:
        config.datasetf = [data_config_dict['train_file'], data_config_dict['valid_file']]
    
    return config


def load_config_with_base(config_name: str, configs_dir: str = "configs") -> Config:
    """
    Load configuration with automatic base config merging
    
    Args:
        config_name: Name of the config (e.g., 'graphfix34')
        configs_dir: Directory containing config files
        
    Returns:
        Config object
    """
    configs_path = Path(configs_dir)
    base_config_path = configs_path / "base.yaml"
    specific_config_path = configs_path / f"{config_name}.yaml"
    
    return load_config(
        str(specific_config_path),
        str(base_config_path) if base_config_path.exists() else None
    )

