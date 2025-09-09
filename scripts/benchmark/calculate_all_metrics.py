#!/usr/bin/env python
"""
Calculate AUROC, EF1, EF5, and BEDROC metrics for all targets across different ablation types.
Results are saved as CSV files in {ablation_type}/analysis/ directories.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from sklearn.metrics import roc_auc_score
except ImportError:
    print("Error: sklearn not found. Install with: pip install scikit-learn")
    sys.exit(1)

# Configuration
MIN_ACTIVES_FOR_METRICS = 10
DEFAULT_DATA_DIR = '/home/j2ho/projects/vs_benchmarks/data'

def read_decoy_data(decoy_file):
    """Read the decoy data file and parse actives/decoys with relationships"""
    actives = set()
    decoys = set()
    active_to_decoys = {}
    
    with open(decoy_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and ';' in line:
                active, decoys_str = line.split(';', 1)
                active = active.strip()
                actives.add(active)
                
                decoy_list = [d.strip() for d in decoys_str.split(',')]
                decoys.update(decoy_list)
                active_to_decoys[active] = decoy_list
    
    return actives, decoys, active_to_decoys

def extract_chembl_id(compound_name):
    """Extract CHEMBL ID from compound name"""
    match = re.search(r'CHEMBL\d+', compound_name)
    if match:
        return match.group()
    return compound_name

def calculate_enrichment_factor(y_true, y_scores, fraction):
    """Calculate enrichment factor at given fraction"""
    if len(y_true) == 0:
        return 0.0
    
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_y_true = np.array(y_true)[sorted_indices]
    
    n_total = len(y_true)
    n_selected = int(fraction * n_total)
    
    if n_selected == 0:
        return 0.0
    
    n_actives_selected = np.sum(sorted_y_true[:n_selected])
    n_actives_total = np.sum(y_true)
    
    if n_actives_total == 0:
        return 0.0
    
    expected_rate = n_actives_total / n_total
    actual_rate = n_actives_selected / n_selected
    
    return actual_rate / expected_rate if expected_rate > 0 else 0.0

def compute_bedroc(y_true, y_scores, alpha=20.0):
    """Compute BEDROC score"""
    if len(y_true) == 0 or np.sum(y_true) == 0:
        return np.nan
    
    sorted_indices = np.argsort(y_scores)[::-1]
    sorted_y_true = np.array(y_true)[sorted_indices]
    
    n_total = len(y_true)
    n_actives = np.sum(y_true)
    
    ranks = np.where(sorted_y_true == 1)[0]
    ra = ranks / n_total
    
    exp_terms = np.exp(-alpha * ra)
    sum_exp = exp_terms.sum()
    
    factor = alpha / (1 - np.exp(-alpha))
    bedroc = sum_exp * factor / n_total
    
    ideal = n_actives * factor / n_total
    
    return bedroc / ideal if ideal > 0 else np.nan

def load_target_results(target_dir: str, target_id: str) -> Optional[pd.DataFrame]:
    """Load all binding predictions for a specific target"""
    patterns = [
        f"{target_dir}/**/binding_predictions.csv",
        f"{target_dir}/*_binding_predictions.csv", 
        f"{target_dir}/binding_predictions.csv"
    ]
    
    result_files = []
    for pattern in patterns:
        result_files.extend(glob.glob(pattern, recursive=True))
    
    if not result_files:
        return None
    
    all_results = []
    for result_file in result_files:
        try:
            df = pd.read_csv(result_file)
            df['chembl_id'] = df['compound'].apply(extract_chembl_id)
            all_results.append(df)
        except Exception as e:
            print(f"    Error reading {result_file}: {e}")
            continue
    
    if not all_results:
        return None
    
    # Combine results and keep best score for each compound
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df = combined_df.sort_values('binding_score', ascending=False)
    combined_df = combined_df.drop_duplicates(subset=['chembl_id'], keep='first')
    
    return combined_df

def apply_exclusion_logic(results_df, actives, decoys, active_to_decoys):
    """Apply exclusion logic for missing actives and invalid scores"""
    if results_df is None or results_df.empty or not actives or not active_to_decoys:
        return [], 0, 0, 0, 0
    
    # Filter valid scores
    valid_scores = results_df.dropna(subset=['binding_score'])
    
    # Create molecule scores dictionary
    molecule_scores = {}
    for _, row in valid_scores.iterrows():
        chembl_id = row['chembl_id']
        score = row['binding_score']
        
        if chembl_id not in molecule_scores:
            molecule_scores[chembl_id] = {
                'chembl_id': chembl_id,
                'score': score,
                'compound': row['compound']
            }
    
    # Find actives with scores
    actives_with_scores = set()
    for chembl_id in molecule_scores.keys():
        if chembl_id in actives:
            actives_with_scores.add(chembl_id)
    
    # Exclude decoys for actives without scores
    actives_without_scores = actives - actives_with_scores
    decoys_to_exclude = set()
    
    for active_without_score in actives_without_scores:
        if active_without_score in active_to_decoys:
            corresponding_decoys = active_to_decoys[active_without_score]
            decoys_to_exclude.update(corresponding_decoys)
    
    # Filter results
    filtered_results = []
    actives_count = 0
    decoys_count = 0
    excluded_actives = len(actives_without_scores)
    excluded_decoys = 0
    
    for chembl_id, result in molecule_scores.items():
        if chembl_id in actives and chembl_id in actives_with_scores:
            result['active_or_decoy'] = 1
            result['label'] = 'active'
            filtered_results.append(result)
            actives_count += 1
        elif chembl_id in decoys and chembl_id not in decoys_to_exclude:
            result['active_or_decoy'] = 0
            result['label'] = 'decoy'
            filtered_results.append(result)
            decoys_count += 1
        elif chembl_id in decoys and chembl_id in decoys_to_exclude:
            excluded_decoys += 1
    
    return filtered_results, actives_count, decoys_count, excluded_actives, excluded_decoys

def calculate_metrics(filtered_results):
    """Calculate AUROC, EF1, EF5, and BEDROC metrics"""
    if not filtered_results:
        return None, "No results"
    
    y_true = [r['active_or_decoy'] for r in filtered_results]
    y_scores = [r['score'] for r in filtered_results]
    
    n_actives = sum(y_true)
    n_decoys = len(y_true) - n_actives
    
    if n_actives < MIN_ACTIVES_FOR_METRICS:
        return None, f"Insufficient actives ({n_actives} < {MIN_ACTIVES_FOR_METRICS})"
    
    if len(set(y_true)) < 2:
        return None, "Need both actives and decoys"
    
    try:
        auroc = roc_auc_score(y_true, y_scores)
        ef_1 = calculate_enrichment_factor(y_true, y_scores, 0.01)
        ef_5 = calculate_enrichment_factor(y_true, y_scores, 0.05)
        bedroc = compute_bedroc(y_true, y_scores, alpha=20.0)
        
        return {
            'AUROC': auroc,
            'EF_1': ef_1,
            'EF_5': ef_5,
            'BEDROC': bedroc,
            'n_actives': n_actives,
            'n_decoys': n_decoys,
            'n_total': len(y_true)
        }, "Success"
    except Exception as e:
        return None, f"Error calculating metrics: {e}"

def save_combined_csv(filtered_results, target_dir, target_id):
    """Save combined CSV with all compounds and their scores/labels for a target"""
    if not filtered_results:
        return
    
    # Create DataFrame from filtered results
    csv_data = []
    for result in filtered_results:
        csv_data.append({
            'compound': result['compound'],
            'chembl_id': result['chembl_id'],
            'binding_score': result['score'],
            'label': result['label'],  # 'active' or 'decoy'
            'active_or_decoy': result['active_or_decoy']  # 1 or 0
        })
    
    df = pd.DataFrame(csv_data)
    # Sort by binding score descending
    df = df.sort_values('binding_score', ascending=False)
    
    # Save to target directory
    os.makedirs(target_dir, exist_ok=True)
    output_file = os.path.join(target_dir, f'{target_id}_combined_results.csv')
    df.to_csv(output_file, index=False)
    print(f"    Saved combined results to: {output_file}")

def process_target(target_id, results_dir, data_dir):
    """Process a single target and calculate metrics"""
    print(f"  Processing target: {target_id}")
    
    # Check for decoy file
    decoy_file = os.path.join(data_dir, target_id, 'decoy_per_active_d3.csv')
    if not os.path.exists(decoy_file):
        print(f"    No decoy file found: {decoy_file}")
        return None
    
    # Read decoy relationships
    actives, decoys, active_to_decoys = read_decoy_data(decoy_file)
    
    # Load results
    target_dir = os.path.join(results_dir, target_id)
    results_df = load_target_results(target_dir, target_id)
    
    if results_df is None or len(results_df) == 0:
        print(f"    No results found for {target_id}")
        return None
    
    # Apply exclusion logic
    filtered_results, actives_count, decoys_count, excluded_actives, excluded_decoys = apply_exclusion_logic(
        results_df, actives, decoys, active_to_decoys
    )
    
    # Save combined CSV for this target
    save_combined_csv(filtered_results, target_dir, target_id)
    
    # Calculate metrics
    metrics, status_msg = calculate_metrics(filtered_results)
    
    result = {
        'target_id': target_id,
        'total_actives': len(actives),
        'total_decoys': len(decoys),
        'actives_with_score': actives_count,
        'decoys_with_score': decoys_count,
        'excluded_actives': excluded_actives,
        'excluded_decoys': excluded_decoys,
        'status': status_msg
    }
    
    if metrics:
        result.update(metrics)
        print(f"    AUROC: {metrics['AUROC']:.3f}, EF1: {metrics['EF_1']:.2f}, EF5: {metrics['EF_5']:.2f}, BEDROC: {metrics['BEDROC']:.3f}")
    else:
        print(f"    Metrics not calculated: {status_msg}")
    
    return result

def process_ablation_type(ablation_dir, data_dir):
    """Process all targets for a specific ablation type"""
    ablation_type = os.path.basename(ablation_dir)
    print(f"\nProcessing ablation type: {ablation_type}")
    
    results_dir = os.path.join(ablation_dir, 'results')
    if not os.path.exists(results_dir):
        print(f"  No results directory found: {results_dir}")
        return
    
    # Find all target directories
    target_dirs = [d for d in os.listdir(results_dir) 
                  if os.path.isdir(os.path.join(results_dir, d))]
    
    if not target_dirs:
        print(f"  No target directories found in {results_dir}")
        return
    
    print(f"  Found {len(target_dirs)} targets")
    
    # Process all targets
    all_results = []
    for target_id in sorted(target_dirs):
        result = process_target(target_id, results_dir, data_dir)
        if result:
            all_results.append(result)
    
    if not all_results:
        print(f"  No valid results for {ablation_type}")
        return
    
    # Create analysis directory
    analysis_dir = os.path.join(ablation_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    output_file = os.path.join(analysis_dir, f'{ablation_type}_metrics.csv')
    results_df.to_csv(output_file, index=False)
    
    # Print summary
    successful_metrics = results_df[results_df['status'] == 'Success']
    if len(successful_metrics) > 0:
        print(f"  Saved metrics for {len(successful_metrics)} targets to: {output_file}")
        print(f"  Mean AUROC: {successful_metrics['AUROC'].mean():.3f} ± {successful_metrics['AUROC'].std():.3f}")
        print(f"  Mean EF1: {successful_metrics['EF_1'].mean():.2f} ± {successful_metrics['EF_1'].std():.2f}")
        print(f"  Mean EF5: {successful_metrics['EF_5'].mean():.2f} ± {successful_metrics['EF_5'].std():.2f}")
        print(f"  Mean BEDROC: {successful_metrics['BEDROC'].mean():.3f} ± {successful_metrics['BEDROC'].std():.3f}")
    else:
        print(f"  No successful metrics calculated for {ablation_type}")

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics for all ablation types')
    parser.add_argument('--root_dir', type=str, default='.',
                       help='Root directory containing ablation type directories (default: current directory)')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                       help='Directory containing decoy files')
    parser.add_argument('--ablation_types', type=str, nargs='+',
                       help='Specific ablation types to process (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Find ablation type directories
    if args.ablation_types:
        ablation_dirs = [os.path.join(args.root_dir, abl) for abl in args.ablation_types
                        if os.path.exists(os.path.join(args.root_dir, abl, 'results'))]
    else:
        # Auto-detect directories with results subdirectory
        ablation_dirs = []
        for item in os.listdir(args.root_dir):
            item_path = os.path.join(args.root_dir, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'results')):
                ablation_dirs.append(item_path)
    
    if not ablation_dirs:
        print("No ablation type directories with results found!")
        print(f"Searched in: {args.root_dir}")
        return
    
    print(f"Found {len(ablation_dirs)} ablation type directories to process:")
    for abl_dir in ablation_dirs:
        print(f"  - {os.path.basename(abl_dir)}")
    
    # Process each ablation type
    for ablation_dir in ablation_dirs:
        process_ablation_type(ablation_dir, args.data_dir)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()