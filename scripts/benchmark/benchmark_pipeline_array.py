#!/usr/bin/env python
"""
Benchmark Pipeline for MotifScreen-Aff
Runs inference on multiple targets from CSV file
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
import glob
import json
from typing import List, Dict, Tuple, Optional

def load_targets_csv(csv_path: str) -> pd.DataFrame:
    """Load targets from CSV file"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} targets from {csv_path}")
    return df

def find_crystal_ligand(uniprot_id: str, pdbid_chain: str, base_dir: str = '/home/j2ho/projects/vs_benchmarks/data') -> Optional[str]:
    """Find crystal ligand file for a target"""
    crystal_dir = f"{base_dir}/{uniprot_id}/crystal_ligs"
    pattern = f"{crystal_dir}/{pdbid_chain}_*.mol2"
    
    crystal_files = glob.glob(pattern)
    if not crystal_files:
        print(f"Warning: No crystal ligand found for {uniprot_id}/{pdbid_chain} in {crystal_dir}")
        return None
    
    # Return the first one found (alphabetically)
    crystal_file = sorted(crystal_files)[0]
    print(f"Found crystal ligand: {crystal_file}")
    return crystal_file

def find_ligands_files(uniprot_id: str, ligands_base_dir: str = '/home/j2ho/DB/ChEMBL/chembl_34/PDBNR') -> List[str]:
    """Find all ligands files for a target"""
    ligands_dir = f"{ligands_base_dir}/{uniprot_id}/batch_mol2s_qh"
    pattern = f"{ligands_dir}/CHEMBL*_b.mol2"
    
    ligands_files = glob.glob(pattern)
    if not ligands_files:
        print(f"Warning: No ligands files found for {uniprot_id} in {ligands_dir}")
        return []
    
    print(f"Found {len(ligands_files)} ligands files for {uniprot_id}")
    return sorted(ligands_files)

def validate_target_files(uniprot_id: str, pdbid_chain: str, 
                         protein_base_dir: str = '/home/j2ho/projects/vs_benchmarks/data') -> Tuple[bool, Dict]:
    """Validate that all required files exist for a target"""
    files = {}
    
    # Check protein PDB
    protein_pdb = f"{protein_base_dir}/{uniprot_id}/{pdbid_chain}.pdb"
    files['protein_pdb'] = protein_pdb
    if not os.path.exists(protein_pdb):
        print(f"Missing protein PDB: {protein_pdb}")
        return False, files
    
    # Check crystal ligand
    crystal_ligand = find_crystal_ligand(uniprot_id, pdbid_chain, protein_base_dir)
    files['crystal_ligand'] = crystal_ligand
    if not crystal_ligand:
        return False, files
    
    # Check ligands files
    ligands_files = find_ligands_files(uniprot_id)
    files['ligands_files'] = ligands_files
    if not ligands_files:
        return False, files
    
    return True, files

def create_shared_protein_files(uniprot_id: str, pdbid_chain: str, crystal_ligand: str,
                               protein_pdb: str, shared_dir: str) -> Tuple[str, str]:
    """Create shared protein files (.prop.npz and .grid.npz) for a target"""
    import subprocess
    import tempfile
    
    # Create shared directory for this protein target
    target_shared_dir = os.path.join(shared_dir, f"{uniprot_id}_{pdbid_chain}")
    os.makedirs(target_shared_dir, exist_ok=True)
    
    prop_file = os.path.join(target_shared_dir, f"{uniprot_id}_{pdbid_chain}.prop.npz")
    grid_file = os.path.join(target_shared_dir, f"{uniprot_id}_{pdbid_chain}.grid.npz")
    
    # Check if files already exist
    if os.path.exists(prop_file) and os.path.exists(grid_file):
        print(f"  Reusing existing protein files for {uniprot_id}/{pdbid_chain}")
        return prop_file, grid_file
    
    print(f"  Creating shared protein files for {uniprot_id}/{pdbid_chain}")
    
    # Use the protein featurizer to create shared files
    sys.path.append('/data/galaxy4/user/j2ho/projects/MotifScreen-Aff')
    from src.io.protein_featurizer import runner as protein_runner
    
    runner_config = {
        'protein_pdb': protein_pdb,
        'crystal_ligand': crystal_ligand,
        'gridsize': 1.5,
        'padding': 10.0,
        'clash': 1.1,
        'output_dir': target_shared_dir
    }
    
    try:
        output_prefix = protein_runner(runner_config)
        created_prop = f"{output_prefix}.prop.npz"
        created_grid = f"{output_prefix}.grid.npz"
        
        # Move to standardized names if different
        if created_prop != prop_file:
            os.rename(created_prop, prop_file)
        if created_grid != grid_file:
            os.rename(created_grid, grid_file)
            
        print(f"  Created: {prop_file}")
        print(f"  Created: {grid_file}")
        
        return prop_file, grid_file
        
    except Exception as e:
        print(f"Error creating protein files for {uniprot_id}/{pdbid_chain}: {e}")
        raise

def create_inference_config(uniprot_id: str, pdbid_chain: str, ligands_file: str, 
                           model_path: str, model_config: str, output_dir: str,
                           shared_prop_file: str, shared_grid_file: str, protein_pdb: str, 
                           crystal_ligand: str, ligands_index: int = 0) -> str:
    """Create inference configuration file for a target"""

    import yaml

    config = {
        'ligands_file': ligands_file,
        'model_path': model_path,
        'output_dir': output_dir,
        'save_aux': False,
        'config_name': model_config,
        'batch_size': 5,
        'protein_pdb': protein_pdb,
        'crystal_ligand': crystal_ligand,
        'shared_prop_file': shared_prop_file,
        'shared_grid_file': shared_grid_file,
        'skip_protein_processing': True
    }

    # Config filename
    config_filename = f"inference_config_{uniprot_id}_{pdbid_chain}_{ligands_index:02d}.yaml"

    # Always save in TOP-LEVEL configs directory (not results/configs/)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(output_dir)))  # output_dir = .../results/...
    top_configs_dir = os.path.join(base_dir, "configs")
    os.makedirs(top_configs_dir, exist_ok=True)

    config_path = os.path.join(top_configs_dir, config_filename)

    # Ensure result output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write YAML config
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return config_path

def create_slurm_job_script(uniprot_id: str, pdbid_chain: str, config_path: str, 
                           model_name: str, ligands_index: int, 
                           output_dir: str, base_script_dir: str) -> str:
    """Create SLURM job script for a target with structured directory"""
    job_name = f"{model_name}_{uniprot_id}_{pdbid_chain}_{ligands_index:02d}"
    
    # Create structured script directory: scripts/{uniprot_id}/{model_name}/
    target_script_dir = os.path.join(base_script_dir, uniprot_id, model_name)
    os.makedirs(target_script_dir, exist_ok=True)
    
    # Create structured log directory: scripts/{uniprot_id}/{model_name}/logs/
    log_dir = os.path.join(target_script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    script_content = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH -p gpu-micro.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -o {log_dir}/{job_name}.out
#SBATCH -e {log_dir}/{job_name}.err
#SBATCH --nice=10000

# Environment setup
cd /data/galaxy4/user/j2ho/projects/MotifScreen-Aff

# Avoid port conflicts
export MASTER_PORT=$((12000 + RANDOM % 1000))
export MASTER_ADDR=127.0.0.1

# Run inference
echo "Starting inference for {uniprot_id}/{pdbid_chain} (ligands file {ligands_index})"
echo "Config: {config_path}"
echo "Model: {model_name}"
echo "Target: {uniprot_id}"
echo "Script dir: {target_script_dir}"

python run_motifscreen_unified.py \\
    --mode inference \\
    --inference_config {config_path}

echo "Inference completed for {job_name}"
"""
    
    script_filename = f"job_{job_name}.sh"
    script_path = os.path.join(target_script_dir, script_filename)
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path

def process_inference_results(result_dir: str, uniprot_id: str, pdbid_chain: str, 
                             ligands_index: int) -> Optional[pd.DataFrame]:
    """Process inference results and extract binding scores"""
    results_file = os.path.join(result_dir, "binding_predictions.csv")
    
    if not os.path.exists(results_file):
        print(f"Warning: Results file not found: {results_file}")
        return None
    
    try:
        df = pd.read_csv(results_file)
        df['uniprot_id'] = uniprot_id
        df['pdbid_chain'] = pdbid_chain  
        df['ligands_index'] = ligands_index
        return df
    except Exception as e:
        print(f"Error reading results file {results_file}: {e}")
        return None

def combine_and_save_target_results(output_base_dir: str, uniprot_id: str, pdbid_chain: str, 
                                   model_name: str, per_target_dir: str) -> Optional[pd.DataFrame]:
    """Combine results from multiple ligands files for a single target and save per-target"""
    all_results = []
    
    # Find all result directories for this target (now organized under uniprot_id parent directory)
    pattern = f"{output_base_dir}/{uniprot_id}/{model_name}_{uniprot_id}_{pdbid_chain}_*"
    result_dirs = glob.glob(pattern)
    
    for result_dir in result_dirs:
        # Extract ligands index from directory name
        dir_name = os.path.basename(result_dir)
        try:
            ligands_index = int(dir_name.split('_')[-1])
        except:
            ligands_index = 0
        
        df = process_inference_results(result_dir, uniprot_id, pdbid_chain, ligands_index)
        if df is not None:
            all_results.append(df)
    
    if not all_results:
        print(f"No valid results found for {uniprot_id}/{pdbid_chain}")
        return None
    
    # Combine all results and remove duplicates (keep best score)
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df.sort_values('binding_score', ascending=False)
    combined_df = combined_df.drop_duplicates(subset=['compound'], keep='first')
    
    # Create per-target directory structure
    target_dir = os.path.join(per_target_dir, uniprot_id)
    os.makedirs(target_dir, exist_ok=True)
    
    # Save per-target results
    target_file = os.path.join(target_dir, f'{uniprot_id}_binding_predictions.csv')
    combined_df.to_csv(target_file, index=False)
    
    print(f"  Saved target results: {target_file} ({len(combined_df)} compounds)")
    
    return combined_df
def main():
    parser = argparse.ArgumentParser(description="Benchmark Pipeline (Array job with chunking)")
    parser.add_argument('--targets_csv', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='benchmark_results')
    parser.add_argument('--mode', type=str, choices=['prepare', 'collect'], default='prepare')
    parser.add_argument('--max_targets', type=int)
    parser.add_argument('--protein_base_dir', type=str,
                        default='/home/j2ho/projects/vs_benchmarks/data')
    parser.add_argument('--ligands_base_dir', type=str,
                        default='/home/j2ho/DB/ChEMBL/chembl_34/PDBNR')
    parser.add_argument('--max_array_size', type=int, default=5000,
                        help='Maximum number of array tasks per job (split if exceeded)')
    parser.add_argument('--concurrency', type=int, default=200,
                        help='Max concurrent tasks per array job (%N)')
    args = parser.parse_args()

    targets_df = load_targets_csv(args.targets_csv)
    if args.max_targets:
        targets_df = targets_df.head(args.max_targets)

    output_base_dir = args.output_dir
    configs_dir = os.path.join(output_base_dir, "configs")
    scripts_dir = os.path.join(output_base_dir, "scripts")
    results_dir = os.path.join(output_base_dir, "results")
    logs_dir = os.path.join(output_base_dir, "logs")

    os.makedirs(configs_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    if args.mode == "prepare":
        triplets = []
        valid_targets = 0
        num_configs = 0

        shared_dir = os.path.join(output_base_dir, "shared_protein_files")
        os.makedirs(shared_dir, exist_ok=True)

        for _, row in targets_df.iterrows():
            uniprot_id = row['uniprot_ID']
            pdbid_chain = row['PDBID_CHAIN']

            files_valid, target_files = validate_target_files(
                uniprot_id, pdbid_chain, args.protein_base_dir
            )
            if not files_valid:
                continue
            if len(target_files['ligands_files']) < 5:
                continue

            try:
                shared_prop_file, shared_grid_file = create_shared_protein_files(
                    uniprot_id, pdbid_chain, target_files['crystal_ligand'],
                    target_files['protein_pdb'], shared_dir
                )
            except Exception as e:
                print(f"Protein processing failed for {uniprot_id}/{pdbid_chain}: {e}")
                continue

            for ligands_idx, ligands_file in enumerate(target_files['ligands_files']):
                target_results_dir = os.path.join(results_dir, uniprot_id)
                target_output_dir = os.path.join(
                    target_results_dir,
                    f"{args.model_name}_{uniprot_id}_{pdbid_chain}_{ligands_idx:02d}"
                )
                config_path = create_inference_config(
                    uniprot_id, pdbid_chain, ligands_file,
                    args.model_path, args.model_config, target_output_dir,
                    shared_prop_file, shared_grid_file,
                    target_files['protein_pdb'], target_files['crystal_ligand'],
                    ligands_idx
                )
                triplets.append((uniprot_id, pdbid_chain, ligands_idx))
                num_configs += 1

            valid_targets += 1
        num_configs = len(triplets)
        if num_configs == 0:
            print("No configs generated. Check input CSV or target files.")
            sys.exit(1)

        # === Job Array Chunking ===
        max_array_size = args.max_array_size
        chunks = [triplets[i:i+max_array_size] for i in range(0, num_configs, max_array_size)]
        array_jobs = []

        for chunk_idx, chunk in enumerate(chunks, start=1):
            triplets_path = os.path.join(configs_dir, f"triplets_part{chunk_idx}.txt")
            with open(triplets_path, "w") as f:
                for t in chunk:
                    f.write(f"{t[0]},{t[1]},{t[2]:02d}\n")

            array_job_path = os.path.join(scripts_dir, f"array_job_part{chunk_idx}.sh")
            num_tasks = len(chunk)  # 이 파트의 실제 라인 수

            array_job = f"""#!/bin/bash
#SBATCH --job-name=bench_array_{args.model_name}_p{chunk_idx}
#SBATCH -p gpu-micro.q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH -o {logs_dir}/tmp/array_p{chunk_idx}_%A_%a.out
#SBATCH -e {logs_dir}/tmp/array_p{chunk_idx}_%A_%a.err
#SBATCH --array=1-{num_tasks}%{args.concurrency}
#SBATCH --nice=10000

cd /data/galaxy4/user/j2ho/projects/MotifScreen-Aff
export MASTER_PORT=$((12000 + RANDOM % 1000))
export MASTER_ADDR=127.0.0.1

LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "{triplets_path}")
TARGET=$(echo $LINE | cut -d, -f1)
PDB=$(echo $LINE | cut -d, -f2)
LIGAND_IDX=$(echo $LINE | cut -d, -f3)

CONFIG="{configs_dir}/inference_config_${{TARGET}}_${{PDB}}_${{LIGAND_IDX}}.yaml"

echo "=== Starting inference for $TARGET / $PDB / $LIGAND_IDX ==="
python run_motifscreen_unified.py --mode inference --inference_config "$CONFIG"
echo "=== Done $TARGET / $PDB / $LIGAND_IDX ==="
RAW_DIR="test_benchmark_results/logs/tmp"
mkdir -p "$RAW_DIR"
RAW_OUT="{logs_dir}/tmp/array_p{chunk_idx}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.out"
RAW_ERR="{logs_dir}/tmp/array_p{chunk_idx}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}.err"

LOG_DIR="{logs_dir}/${{TARGET}}/${{PDB}}"
mkdir -p "$LOG_DIR"
mv "$RAW_OUT" "$LOG_DIR/${{LIGAND_IDX}}.out"
mv "$RAW_ERR" "$LOG_DIR/${{LIGAND_IDX}}.err"
"""
            with open(array_job_path, "w") as f:
                f.write(array_job)
            os.chmod(array_job_path, 0o755)
            array_jobs.append(array_job_path)

        # === Master submit_all.sh ===
        submit_script = os.path.join(scripts_dir, "submit_all.sh")
        with open(submit_script, "w") as f:
            f.write("#!/bin/bash\n\n")
            for aj in array_jobs:
                f.write(f"sbatch {aj}\n")
        os.chmod(submit_script, 0o755)

        print(f"Prepared {num_configs} configs, split into {len(chunks)} array jobs")
        print(f"Submit with: {submit_script}")


    elif args.mode == "collect":
        # 기존 collect 로직 그대로 사용
        per_target_dir = os.path.join(output_base_dir, f"{args.model_name}_per_target_results")
        os.makedirs(per_target_dir, exist_ok=True)

        all_results = []
        for _, row in targets_df.iterrows():
            uniprot_id = row['uniprot_ID']
            pdbid_chain = row['PDBID_CHAIN']
            res = combine_and_save_target_results(results_dir, uniprot_id, pdbid_chain,
                                                  args.model_name, per_target_dir)
            if res is not None:
                all_results.append(res)

        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
            out_csv = os.path.join(output_base_dir, f"{args.model_name}_benchmark_results.csv")
            final_df.to_csv(out_csv, index=False)
            print(f"Collected results into {out_csv}")


if __name__ == "__main__":
    main()