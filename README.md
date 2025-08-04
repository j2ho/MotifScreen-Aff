# MotifScreen-Aff Inference Pipeline

This document describes how to use the MotifScreen-Aff inference pipeline for virtual screening and binding prediction.

## Overview

The inference pipeline allows users to:
1. **Screen ligand libraries** against protein targets
2. **Predict binding affinities** and rank compounds  
3. **Identify binding motifs** in the binding pocket
4. **Predict ligand coordinates** in the binding site

## Quick Start

1. **Prepare your config file** (see `example_inference_config.yaml`)
2. **Run inference**:
   ```bash
   python -m run_motifscreen --config your_config.yaml
   ```
   Example run command (supports CPU run):
   ```bash
   python -m run_motifscreen --config configs/example_inference_config.yaml
   ```

## Required Inputs

### 1. Protein PDB File
- Standard PDB format
- Should contain the target protein structure
- Can include crystal waters and other cofactors ??? 수정필요 

### 2. Binding Center
Choose **one** of the following options:

**Option A: Explicit Coordinates**
```yaml
center: [10.5, 15.2, 8.7]  
```

**Option B: Crystal Ligand File**
```yaml
crystal_ligand: "/path/to/crystal_ligand.pdb"   # PDB 
# or
crystal_ligand: "/path/to/crystal_ligand.mol2"  # MOL2 or PDB 
```
The pipeline will automatically calculate the center of mass from the ligand coordinates.

**Priority**: If both `center` and `crystal_ligand` are provided, `center` takes priority.

### 3. Ligands File
**ONLY supports MOL2 Format:**
```mol2
@<TRIPOS>MOLECULE
compound_1
...
@<TRIPOS>MOLECULE  
compound_2
...
```

### 4. Trained Model
- Path to a trained MotifScreen-Aff model checkpoint (`.pkl` file)
- Should be compatible with the specified config

## Configuration File

Create a YAML configuration file based on `example_inference_config.yaml`:

```yaml
# Required parameters
protein_pdb: "/absolute/path/to/protein.pdb"
center: [10.5, 15.2, 8.7]  # Option 1: explicit coordinates
# crystal_ligand: "/path/to/crystal_ligand.pdb"  # Option 2: crystal ligand file
ligands_file: "/absolute/path/to/ligands.mol2"
model_path: "/absolute/path/to/model.pkl"
output_dir: "./results"

# Optional parameters  - user won't change these unless they know what they're doing
gridsize: 1.5      # Grid spacing (Angstroms)
padding: 10.0      # Padding around center (Angstroms)
clash: 1.1         # Clash distance for grid filtering
config_name: "common"  # Model configuration
```

## Pipeline Steps

The inference pipeline consists of 6 main steps:

### Step 1: Protein Processing
- Reads protein PDB file
- Extracts amino acid properties, atom types, charges
- Calculates SASA and other features
- Saves as `target.prop.npz`

### Step 2: Grid Generation
- Generates 3D grid points around binding center
- Filters out clashing and distant points
- Connects largest grid cluster
- Saves as `target.grid.npz`

### Step 3: Ligand Processing
- Reads ligands from MOL2/PDB file
- Performs BRICS fragmentation
- Identifies key atoms for each ligand
- Saves as `keyatom.def.npz`

### Step 4: Model Loading
- Loads trained MotifScreen-Aff model
- Handles DDP checkpoints automatically
- Sets model to evaluation mode

### Step 5: Inference
- Creates molecular graphs for receptor and ligands
- Runs forward pass through the model
- Predicts binding scores, motifs, and coordinates

### Step 6: Results Saving
- Saves complete results as JSON
- Creates ranked CSV file for binding predictions
- Saves motif and coordinate predictions as NumPy arrays

## Output Files

The pipeline generates several output files in the specified `output_dir`:

### Intermediate Files
- `target.prop.npz`: Receptor features and properties
- `target.grid.npz`: Grid points around binding center
- `keyatom.def.npz`: Key atoms extracted from ligands

### Results Files
- `inference_results.json`: Complete results in JSON format
- `binding_predictions.csv`: Ranked binding predictions
- `motif_predictions.npy`: Motif predictions (grid-based)
- `predicted_coordinates.npy`: Predicted ligand coordinates

## Results Interpretation

### Binding Predictions (`binding_predictions.csv`)
```csv
ligand_name,binding_score,rank  
compound_123,0.856743,1
compound_089,0.742156,2
compound_234,0.698432,3
...
```
- **binding_score**: Predicted binding probability (0-1)
- **rank**: Ranking based on binding score (1 = highest)

### Motif Predictions (`motif_predictions.npy`)
- 3D array of shape `(grid_points, motif_types)`
- Motif types: [None, Both, Acceptor, Donor, Aliphatic, Aromatic]
- Values represent predicted probability for each motif type at each grid point

### Coordinate Predictions (`predicted_coordinates.npy`)
- Predicted 3D coordinates of key ligand atoms
- Shape depends on model architecture and number of key atoms

## Example Usage

### Basic Screening
```bash
# 1. Create config file
cat > screening_config.yaml << EOF
protein_pdb: "data/example/receptor.pdb"
center: [10.5, 15.2, 8.7]
ligands_file: "data/example/actives_final.mol2"
model_path: "models/msk_v2.pkl"
output_dir: "./virtual_screening"
EOF

# 2. Run inference
python -m run_motifscreen --config screening_config.yaml
```

### Using Crystal Ligand for Center
```yaml
protein_pdb: "data/example/receptor.pdb"
crystal_ligand: "data/example/crystal_ligand.mol2"
ligands_file: "data/example/actives_final.mol2"
model_path: "models/msk_v2.pkl"
output_dir: "./virtual_screening"
```


## Performance Tips

1. **Use CUDA**: The pipeline automatically detects and uses GPU if available
2. **Batch size**: Currently fixed at 5 for inference 
3. **Memory**: Large ligand libraries may require more RAM
4. **Preprocessing**: Ligand preprocessing (key atom extraction) can be time-consuming

## Troubleshooting

### Common Issues

**FileNotFoundError**: Check that all input file paths are absolute and exist
```bash
ls -la /absolute/path/to/your/files
```

**CUDA out of memory**: Try using CPU-only by setting:
```bash
export CUDA_VISIBLE_DEVICES=""
```

**No key atoms found**: Check that ligand file format is correct and contains valid molecules

**Model loading fails**: Ensure model checkpoint is compatible with the specified config


## Integration with Training Pipeline

The inference pipeline uses the same core components as training:
- `MolecularLoader` for data loading
- `GraphBuilder` for graph construction  
- Model architectures (`MSK_1`, `MSK_2`)
- Configuration system

This ensures consistency between training and inference.

## Dependencies

Required packages:
- PyTorch
- DGL (Deep Graph Library)
- RDKit
- OpenBabel
- NumPy, SciPy
- PyYAML

See `requirements.txt` for specific versions.

## Citation

If you use this inference pipeline, please cite:
```
[future MotifScreen-Aff paper citation...]
```

## Support

For questions or issues:
1. Check this README and example configs
2. Review error messages and logs
3. Open an issue on the project repository
