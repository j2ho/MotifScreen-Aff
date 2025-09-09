# configs/config.py
import yaml
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

@dataclass
class DataPathsConfig: 
    datapath: str = 'data'
    keyatomf: str = 'keyatom.def.npz'
    decoyf: str = 'decoys.BL2.npz'
    affinityf: Optional[str] = None
    crossreceptorf: Optional[str] = 'crossreceptor.filtered.npz'

@dataclass
class GraphParamsConfig: 
    edgemode: str = 'dist'
    edgek: Tuple[int, int] = (8, 16)
    edgedist: Tuple[float, float] = (2.2, 4.5)
    maxedge: int = 100000
    maxnode: int = 3000
    ball_radius: float = 8.0
    ball_radius_var: float = 0.0 #unuse by default
    firstshell_as_grid: bool = False

@dataclass
class DataProcessingConfig: 
    ntype: int = 6
    max_subset: int = 5
    drop_H: bool = False
    store_memory: bool = False

@dataclass
class DataAugmentationConfig: 
    randomize: float = 0.5
    randomize_grid: float = 0.0
    pert: bool = False

@dataclass
class CrossValidationConfig: 
    load_cross: bool = False
    cross_eval_struct: bool = False
    cross_grid: float = 0.0
    nonnative_struct_weight: float = 0.2
    motif_otf: bool = False


@dataclass
class GridParams:
    """GridFeaturizer parameters"""
    model: str = "se3"  
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
    """LigandFeaturizer parameters"""
    model: str = "gat"
    dropout_rate: float = 0.1
    num_layers: int = 2
    n_heads: int = 4
    num_channels: int = 32
    l0_in_features: int = 15
    l0_out_features: int = 32
    num_edge_features: int = 5
    """LigandModule parameters (for global emb)"""
    n_lig_global_in: int = 19
    n_lig_global_out: int = 4


@dataclass
class TRParams:
    """Trigon/Class/Distance/StructModule parameters"""
    dropout_rate: float = 0.1
    d: int = 64  # Now automatically set to c
    c: int = 64
    n_trigon_lig_layers: int = 3
    n_trigon_key_layers: int = 3
    shared_trigon: bool = False
    normalize_Xform: bool = True
    lig_to_key_attn: bool = True
    trigon_module: str = 'trigon'  # 'optimized' or 'trigon'

@dataclass
class AffModuleParams:
    classification_mode: str = "former_contrast"
    dropoutmode: str = 'none'

@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""
    use_scheduler: bool = True
    scheduler_type: str = "ReduceLROnPlateau"  # Options: "ReduceLROnPlateau", "StepLR", "CosineAnnealingLR", "None"
    # ReduceLROnPlateau params
    factor: float = 0.8
    patience: int = 3
    min_lr: float = 1e-6
    threshold: float = 1e-4
    # StepLR params
    step_size: int = 50
    gamma: float = 0.1
    # CosineAnnealingLR params
    T_max: int = 100

@dataclass
class TrainingParamsConfig: 
    lr: float = 1.0e-4
    max_epoch: int = 500
    debug: bool = False
    ddp: bool = True
    silent: bool = False
    accumulation_steps: int = 2
    load_checkpoint: bool = False 
    amp: bool = False  
    wandb_mode: str = "online"  # Options: "online", "disabled", "offline"
    weight_decay: float = 1e-4  # L2 regularization via optimizer weight decay
    max_param_norm: float = 100.0  # Maximum parameter norm before warning/clipping
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    custom_init: bool = False  # Custom initialization flag, can be used to trigger specific init logic

@dataclass
class DataLoaderParamsConfig: # Consolidated dataloader-related params
    batch_size: int = 1
    num_workers: int = 5
    pin_memory: bool = True
    shuffle: bool = True


@dataclass
class LossWeightsConfig: # Consolidated loss weights
    w_cat: float = 0.05
    w_str: float = 0.2
    w_motif_penalty: float = 1.0e-10
    w_penalty: float = 1.0e-10
    w_contrast: float = 2.0
    w_spread: float = 5.0
    w_screen: float = 0.5
    w_screen_contrast: float = 0.5
    w_screen_ranking: float = 5.0
    w_Dkey: float = 1.0
    screenloss: str = "BCE"
    struct_loss: str = "mse"

@dataclass
class Config:
    modelname: str = "MSK"
    version: str = "v1.0"
    model_note: str = ""  # Additional note for the model, can be used for versioning or description

    # Model
    model_params_grid: GridParams = field(default_factory=GridParams) # Renamed to avoid confusion with GraphConfig
    model_params_ligand: LigandParams = field(default_factory=LigandParams)
    model_params_TR: TRParams = field(default_factory=TRParams)
    model_params_aff: AffModuleParams = field(default_factory=AffModuleParams)

    # Data-related
    paths: DataPathsConfig = field(default_factory=DataPathsConfig)
    graph: GraphParamsConfig = field(default_factory=GraphParamsConfig)
    processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)

    # Training and Dataloader 
    training: TrainingParamsConfig = field(default_factory=TrainingParamsConfig)
    dataloader: DataLoaderParamsConfig = field(default_factory=DataLoaderParamsConfig)
    losses: LossWeightsConfig = field(default_factory=LossWeightsConfig)


    # Main model parameters ... now same for all models
    dropout_rate: float = 0.2 # This can be synchronized using __post_init__

    # Dataset file paths
    train_file: str = "data/common_up.train.txt" # Direct path from yaml config
    valid_file: str = "data/common_up.valid.txt" # Direct path from yaml config

    def __post_init__(self):
        # Sync dropout rates from main config to nested model params
        self.model_params_grid.dropout_rate = self.dropout_rate
        self.model_params_ligand.dropout_rate = self.dropout_rate
        self.model_params_TR.dropout_rate = self.dropout_rate

        # DDP 
        if self.training.ddp:
            self.dataloader.shuffle = False
        if self.training.debug:
            self.dataloader.num_workers = 1

def deep_merge(base_dict: Dict, override_dict: Dict) -> Dict:
    result = base_dict.copy()
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_config(config_path: str, base_config_path: Optional[str] = None) -> Config:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    if base_config_path:
        base_path = Path(base_config_path)
        if base_path.exists():
            with open(base_path, 'r') as f:
                base_dict = yaml.safe_load(f)
            config_dict = deep_merge(base_dict, config_dict)

    # Instantiate the sub-dataclasses from the loaded dictionary sections
    return Config(
        modelname=config_dict.get('model', {}).get('name', 'MSK'),
        version=config_dict.get('model', {}).get('version', 'v1.0'),

        model_params_grid=GridParams(**config_dict.get('model', {}).get('grid', {})),
        model_params_ligand=LigandParams(**config_dict.get('model', {}).get('ligand', {})),
        model_params_TR=TRParams(**config_dict.get('model', {}).get('TR', {})),
        model_params_aff=AffModuleParams(**config_dict.get('model', {}).get('aff', {})),

        paths=DataPathsConfig(
            datapath=config_dict.get('data', {}).get('datapath', 'data'),
            keyatomf=config_dict.get('data', {}).get('keyatomf', 'keyatom.def.npz'),
            decoyf=config_dict.get('data', {}).get('decoyf', 'decoys.BL2.npz'),
            affinityf=config_dict.get('data', {}).get('affinityf', None),
            crossreceptorf=config_dict.get('data', {}).get('crossreceptorf', 'crossreceptor.filtered.npz')
        ),
        graph=GraphParamsConfig(
            edgemode=config_dict.get('data', {}).get('edgemode', 'dist'),
            edgek=tuple(config_dict.get('data', {}).get('edgek', [8, 16])),
            edgedist=tuple(config_dict.get('data', {}).get('edgedist', [2.2, 4.5])),
            maxedge=config_dict.get('data', {}).get('maxedge', 100000),
            maxnode=config_dict.get('data', {}).get('maxnode', 3000),
            ball_radius=config_dict.get('data', {}).get('ball_radius', 8.0),
            ball_radius_var=config_dict.get('data', {}).get('ball_radius_var', 0.0),
            firstshell_as_grid=config_dict.get('misc', {}).get('firstshell_as_grid', False) # Moved from data
        ),
        processing=DataProcessingConfig(
            ntype=config_dict.get('data', {}).get('ntype', 6),
            max_subset=config_dict.get('data', {}).get('max_subset', 5),
            drop_H=config_dict.get('data', {}).get('drop_H', False),
            # store_memory is not in YAML, keep dataclass default or add to YAML if needed
        ),
        augmentation=DataAugmentationConfig(
            randomize=config_dict.get('data', {}).get('randomize', 0.5), # Moved from data
            randomize_grid=config_dict.get('misc', {}).get('randomize_grid', 0.0),
            pert=config_dict.get('misc', {}).get('pert', False)
        ),
        cross_validation=CrossValidationConfig(
            load_cross=config_dict.get('cross_validation', {}).get('load_cross', False),
            cross_eval_struct=config_dict.get('cross_validation', {}).get('cross_eval_struct', False),
            cross_grid=config_dict.get('cross_validation', {}).get('cross_grid', 0.0),
            nonnative_struct_weight=config_dict.get('cross_validation', {}).get('nonnative_struct_weight', 0.2),
            motif_otf=config_dict.get('cross_validation',{}).get('motif_otf', False)
        ),

        training=TrainingParamsConfig(
            lr=config_dict.get('training', {}).get('lr', 1.0e-4),
            max_epoch=config_dict.get('training', {}).get('max_epoch', 500),
            debug=config_dict.get('training', {}).get('debug', False),
            ddp=config_dict.get('training', {}).get('ddp', True),
            silent=config_dict.get('training', {}).get('silent', False),
            accumulation_steps=config_dict.get('training', {}).get('accumulation_steps', 1),
            load_checkpoint=config_dict.get('training', {}).get('load_checkpoint', False),
            amp=config_dict.get('training', {}).get('amp', False), 
            wandb_mode=config_dict.get('training', {}).get('wandb_mode', 'online'),  # Added wandb_mode 
            weight_decay=config_dict.get('training', {}).get('weight_decay', 1e-4),
            max_param_norm=config_dict.get('training', {}).get('max_param_norm', 100.0),
            custom_init=config_dict.get('training', {}).get('custom_init', False),  # Custom initialization flag
            scheduler=SchedulerConfig(
                use_scheduler=config_dict.get('training', {}).get('scheduler', {}).get('use_scheduler', True),
                scheduler_type=config_dict.get('training', {}).get('scheduler', {}).get('scheduler_type', 'ReduceLROnPlateau'),
                factor=config_dict.get('training', {}).get('scheduler', {}).get('factor', 0.8),
                patience=config_dict.get('training', {}).get('scheduler', {}).get('patience', 3),
                min_lr=config_dict.get('training', {}).get('scheduler', {}).get('min_lr', 1e-6),
                threshold=config_dict.get('training', {}).get('scheduler', {}).get('threshold', 1e-4),
                step_size=config_dict.get('training', {}).get('scheduler', {}).get('step_size', 50),
                gamma=config_dict.get('training', {}).get('scheduler', {}).get('gamma', 0.1),
                T_max=config_dict.get('training', {}).get('scheduler', {}).get('T_max', 100)
            )
        ),
        dataloader=DataLoaderParamsConfig(
            batch_size=config_dict.get('dataloader', {}).get('batch_size', 1),
            num_workers=config_dict.get('dataloader', {}).get('num_workers', 5),
            pin_memory=config_dict.get('dataloader', {}).get('pin_memory', True),
            shuffle=config_dict.get('dataloader', {}).get('shuffle', True)
        ),
        losses=LossWeightsConfig(
            w_cat=config_dict.get('losses', {}).get('w_cat', 0.05),
            w_str=config_dict.get('losses', {}).get('w_str', 0.2),
            w_motif_penalty=config_dict.get('losses', {}).get('w_motif_penalty', 1.0e-10),
            w_penalty=config_dict.get('losses', {}).get('w_penalty', 1.0e-10),
            w_contrast=config_dict.get('losses', {}).get('w_contrast', 2.0),
            w_spread=config_dict.get('losses', {}).get('w_spread', 5.0),
            w_screen=config_dict.get('losses', {}).get('w_screen', 0.5),
            w_screen_contrast=config_dict.get('losses', {}).get('w_screen_contrast', 0.5),
            w_screen_ranking=config_dict.get('losses', {}).get('w_screen_ranking', 5.0),
            w_Dkey=config_dict.get('losses', {}).get('w_Dkey', 1.0),
            screenloss=config_dict.get('losses', {}).get('screenloss', "BCE"),
            struct_loss=config_dict.get('losses', {}).get('struct_loss', "mse")
        ),

        # Directly from root or training section
        dropout_rate=config_dict.get('common', {}).get('dropout_rate', 0.2), # Synchronized in __post_init__

        # Dataset file paths
        train_file=config_dict.get('data', {}).get('train_file', "data/PLmix.60k.screen.txt"),
        valid_file=config_dict.get('data', {}).get('valid_file', "data/PLmix.60k.screen.txt")
    )

def load_config_with_base(config_name: str, configs_dir: str = "configs") -> Config:
    configs_path = Path(configs_dir)
    base_config_path = configs_path / "common.yaml"
    specific_config_path = configs_path / f"{config_name}.yaml"
    return load_config(
        str(specific_config_path),
        str(base_config_path) if base_config_path.exists() else None
    )
