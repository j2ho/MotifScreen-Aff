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
    dropout_rate: float = 0.1
    num_layers: int = 2
    n_heads: int = 4
    num_channels: int = 32
    l0_in_features: int = 15
    l0_out_features: int = 32
    num_edge_features: int = 5


@dataclass
class TRParams:
    """TR/SE3 parameters - corresponds to TRArgs.params"""
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


@dataclass
class TrainingConfig:
    """Training configuration"""
    lr: float = 1.0e-4
    max_epoch: int = 500
    debug: bool = False
    ddp: bool = True
    silent: bool = False


@dataclass
class DataConfig:
    """Data configuration"""
    datapath: str = "/ml/motifnet/features_com2/"
    train_file: str = "data/PLmix.60k.screen.txt"
    valid_file: str = "data/PLmix.60k.screen.txt"
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
    modelname: str = "base_model"
    
    # Core parameters
    dropout_rate: float = 0.2
    m: int = 64  # embedding dimension
    
    # Parameter groups (using dataclasses instead of dicts)
    params_grid: GridParams = field(default_factory=GridParams)
    params_ligand: LigandParams = field(default_factory=LigandParams)
    params_TR: TRParams = field(default_factory=TRParams)
    
    # Main model parameters
    classification_mode: str = "former_contrast"
    ntypes: int = 6
    LR: float = 1.0e-4
    wTR: float = 0.2
    wGrid: float = 1.0
    w_reg: float = 1.0e-10
    screenloss: str = "BCE"
    w_contrast: float = 2.0
    w_false: float = 0.2
    w_spread: float = 5.0
    w_screen: float = 0.0
    w_screen_contrast: float = 0.0
    w_screen_ranking: float = 0.0
    w_Dkey: float = 1.0
    trim_receptor_embedding: bool = True
    max_epoch: int = 500
    debug: bool = False
    datasetf: Union[str, List[str]] = "data/PLmix.60k.screen.txt"
    n_lig_feat: int = 19
    n_lig_emb: int = 4
    struct_loss: str = "mse"
    input_features: str = "base"
    pert: bool = False
    ligand_model: str = "gat"
    load_cross: bool = False
    cross_eval_struct: bool = False
    cross_grid: float = 0.0
    nonnative_struct_weight: float = 0.2
    randomize_grid: float = 0.0
    shared_trigon: bool = False
    normalize_Xform: bool = True
    lig_to_key_attn: bool = True
    keyatomf: str = "keyatom.def.npz"
    firstshell_as_grid: bool = False
    use_input_PHcore: bool = False
    
    # Training options
    ddp: bool = True
    silent: bool = False
    
    # Data and loader configs
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)

    def __post_init__(self):
        """Post-initialization to handle parameter adjustments"""
        self._sync_dropout_rates()
        self._sync_embedding_dimensions()
        self._apply_feature_adjustments()
        self._apply_ddp_adjustments()

    def _sync_dropout_rates(self):
        """Sync dropout rates across all parameter groups"""
        self.params_grid.dropout_rate = self.dropout_rate
        self.params_ligand.dropout_rate = self.dropout_rate
        self.params_TR.dropout_rate = self.dropout_rate

    def _sync_embedding_dimensions(self):
        """Sync embedding dimensions across parameter groups"""
        self.params_grid.l0_out_features = self.m
        self.params_ligand.l0_out_features = self.m
        self.params_TR.m = self.m
        self.params_TR.l0_out_features_lig = self.m

    def _apply_feature_adjustments(self):
        """Apply feature-specific parameter adjustments"""
        if self.input_features == "base":
            pass
        elif self.input_features == "ex1":
            self.params_ligand.l0_in_features = 18
            self.params_grid.l0_in_features = 104
        elif self.input_features == "ex2":
            self.n_lig_feat = 32
            self.params_ligand.l0_in_features = 18
            self.params_grid.l0_in_features = 104
        elif self.input_features == "graph":
            self.params_ligand.l0_in_features = 18
            self.params_grid.l0_in_features = 104
        elif self.input_features == "graphex":
            self.params_ligand.l0_in_features = 69
            self.n_lig_feat = 19 + 128
            self.n_lig_emb = 8
            self.params_ligand.num_edge_features = 4
            self.params_grid.l0_in_features = 104
        elif self.input_features == "graphfix":
            self.params_ligand.l0_in_features = 21
            self.params_grid.l0_in_features = 104

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

    def feattype(self, feat: str):
        """Set feature type and apply adjustments"""
        self.input_features = feat
        self._apply_feature_adjustments()


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
    tr_params = TRParams(**config_dict.get('model', {}).get('se3', {}))
    
    data_config = DataConfig(**config_dict.get('data', {}))
    dataloader_config = DataLoaderConfig(**config_dict.get('dataloader', {}))
    
    # Build main config
    main_config = config_dict.get('model', {})
    training_config = config_dict.get('training', {})
    loss_config = config_dict.get('losses', {})
    feature_config = config_dict.get('features', {})
    cv_config = config_dict.get('cross_validation', {})
    misc_config = config_dict.get('misc', {})
    
    # Create config with all parameters
    config = Config(
        # Model name
        modelname=main_config.get('name', 'base_model'),
        
        # Core parameters
        dropout_rate=main_config.get('dropout_rate', 0.2),
        m=main_config.get('m', 64),
        
        # Parameter groups
        params_grid=grid_params,
        params_ligand=ligand_params,
        params_TR=tr_params,
        
        # Main model parameters
        classification_mode=main_config.get('classification_mode', 'former_contrast'),
        ntypes=main_config.get('ntypes', 6),
        LR=training_config.get('lr', 1.0e-4),
        wTR=loss_config.get('wTR', 0.2),
        wGrid=loss_config.get('wGrid', 1.0),
        w_reg=loss_config.get('w_reg', 1.0e-10),
        screenloss=loss_config.get('screenloss', 'BCE'),
        w_contrast=loss_config.get('w_contrast', 2.0),
        w_false=loss_config.get('w_false', 0.2),
        w_spread=loss_config.get('w_spread', 5.0),
        w_screen=loss_config.get('w_screen', 0.0),
        w_screen_contrast=loss_config.get('w_screen_contrast', 0.0),
        w_screen_ranking=loss_config.get('w_screen_ranking', 0.0),
        w_Dkey=loss_config.get('w_Dkey', 1.0),
        trim_receptor_embedding=main_config.get('trim_receptor_embedding', True),
        max_epoch=training_config.get('max_epoch', 500),
        debug=training_config.get('debug', False),
        datasetf=config_dict.get('data', {}).get('train_file', 'data/PLmix.60k.screen.txt'),
        n_lig_feat=feature_config.get('n_lig_feat', 19),
        n_lig_emb=feature_config.get('n_lig_emb', 4),
        struct_loss=loss_config.get('struct_loss', 'mse'),
        input_features=feature_config.get('input_features', 'base'),
        pert=misc_config.get('pert', False),
        ligand_model=main_config.get('ligand', {}).get('model_type', 'gat'),
        load_cross=cv_config.get('load_cross', False),
        cross_eval_struct=cv_config.get('cross_eval_struct', False),
        cross_grid=cv_config.get('cross_grid', 0.0),
        nonnative_struct_weight=cv_config.get('nonnative_struct_weight', 0.2),
        randomize_grid=misc_config.get('randomize_grid', 0.0),
        shared_trigon=main_config.get('se3', {}).get('shared_trigon', False),
        normalize_Xform=main_config.get('se3', {}).get('normalize_Xform', True),
        lig_to_key_attn=main_config.get('se3', {}).get('lig_to_key_attn', True),
        keyatomf=misc_config.get('keyatomf', 'keyatom.def.npz'),
        firstshell_as_grid=misc_config.get('firstshell_as_grid', False),
        use_input_PHcore=misc_config.get('use_input_PHcore', False),
        
        # Training options
        ddp=training_config.get('ddp', True),
        silent=training_config.get('silent', False),
        
        # Data and loader configs
        data=data_config,
        dataloader=dataloader_config
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


# Legacy compatibility functions
def create_common_config() -> Config:
    """Create common MSK config similar to args.py"""
    config = Config(
        modelname="graphfix34",
        dropout_rate=0.2,
        m=64,
        
        # Enhanced architecture
        params_grid=GridParams(
            num_layers_grid=5,
            l0_in_features=104,
            dropout_rate=0.2,
            l0_out_features=64
        ),
        params_ligand=LigandParams(
            num_layers=4,
            l0_in_features=21,
            dropout_rate=0.2,
            l0_out_features=64
        ),
        params_TR=TRParams(
            num_layers_lig=3,
            n_trigon_lig_layers=2,
            n_trigon_key_layers=3,
            dropout_rate=0.2,
            m=64,
            l0_out_features_lig=64
        ),
        
        # Data augmentation
        pert=True,
        load_cross=True,
        cross_eval_struct=True,
        cross_grid=1.0,
        
        # Loss weights
        wGrid=0.05,
        w_screen=0.5,
        w_screen_contrast=0.5,
        w_screen_ranking=5.0,
        
        # Feature settings
        input_features="graph",
        firstshell_as_grid=False,
        
        # Data files
        datasetf=['data/common_up.train.txt', 'data/common_up.valid.txt']
    )
    
    return config