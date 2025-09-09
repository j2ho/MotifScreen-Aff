#!/usr/bin/env python
"""
Analyze MotifScreen-Aff benchmark results and calculate AUROC per target
Uses the same active/decoy logic as universal_screening_analysis.py
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
import json
import re

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.metrics import precision_recall_curve, average_precision_score
except ImportError:
    print("Warning: sklearn not found. Install with: pip install scikit-learn")
    sys.exit(1)

import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
MIN_ACTIVES_FOR_METRICS = 10

def read_decoy_data(decoy_file):
    """Read the decoy data file and parse actives/decoys with relationships (same as reference script)"""
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

def load_target_results(target_dir: str, uniprot_id: str) -> Optional[pd.DataFrame]:
    """Load all MotifScreen results for a specific target"""
    # Try multiple patterns to find the results files
    patterns = [
        f"{target_dir}/**/binding_predictions.csv",  # Original pattern
        f"{target_dir}/*_binding_predictions.csv",   # New pattern: Q10469_binding_predictions.csv
        f"{target_dir}/binding_predictions.csv"      # Direct file
    ]
    
    result_files = []
    for pattern in patterns:
        result_files.extend(glob.glob(pattern, recursive=True))
    
    if not result_files:
        print(f"  No results found for {uniprot_id} in {target_dir}")
        print(f"  Searched patterns: {patterns}")
        return None
    
    all_results = []
    for result_file in result_files:
        try:
            df = pd.read_csv(result_file)
            # Extract CHEMBL IDs from compound names
            df['chembl_id'] = df['compound'].apply(extract_chembl_id)
            all_results.append(df)
        except Exception as e:
            print(f"  Error reading {result_file}: {e}")
            continue
    
    if not all_results:
        return None
    
    # Combine all results and remove duplicates (keep best score)
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Keep highest binding score for each compound
    combined_df = combined_df.sort_values('binding_score', ascending=False)
    combined_df = combined_df.drop_duplicates(subset=['chembl_id'], keep='first')
    
    return combined_df

def apply_enhanced_exclusion(results_df, actives, decoys, active_to_decoys):
    """Apply enhanced exclusion logic for missing actives and NaN scores (same as reference script)"""
    if results_df is None or results_df.empty or not actives or not active_to_decoys:
        return [], 0, 0, 0, 0
    
    # Filter out results with NaN scores
    valid_scores = results_df.dropna(subset=['binding_score'])
    nan_filtered = len(results_df) - len(valid_scores)
    
    if nan_filtered > 0:
        print(f"    Filtered {nan_filtered} entries with NaN/invalid scores")
    
    # Create molecule scores dictionary
    molecule_scores = {}
    for _, row in valid_scores.iterrows():
        chembl_id = row['chembl_id']
        score = row['binding_score']
        
        # Keep best score for each molecule (already sorted by binding_score desc)
        if chembl_id not in molecule_scores:
            molecule_scores[chembl_id] = {
                'chembl_id': chembl_id,
                'score': score,
                'compound': row['compound'],
                'rank': row.get('rank', 0)
            }
    
    # Find actives with valid scores
    actives_with_scores = set()
    for chembl_id in molecule_scores.keys():
        if chembl_id in actives:
            actives_with_scores.add(chembl_id)
    
    # Find actives without scores (either missing entirely or had NaN scores)
    actives_without_scores = actives - actives_with_scores
    decoys_to_exclude = set()
    
    # Exclude decoys corresponding to actives without valid scores
    for active_without_score in actives_without_scores:
        if active_without_score in active_to_decoys:
            corresponding_decoys = active_to_decoys[active_without_score]
            decoys_to_exclude.update(corresponding_decoys)
    
    # Filter results based on enhanced exclusion logic
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

def calculate_enrichment_factor(y_true, y_scores, fraction):
    """Calculate enrichment factor at given fraction (same as reference script)"""
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
    """Compute BEDROC score (same as reference script)"""
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

def calculate_metrics(filtered_results):
    """Calculate performance metrics (same as reference script)"""
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
        # MotifScreen scores are higher=better
        auroc = roc_auc_score(y_true, y_scores)
        ef_1 = calculate_enrichment_factor(y_true, y_scores, 0.01)
        ef_5 = calculate_enrichment_factor(y_true, y_scores, 0.05)
        bedroc = compute_bedroc(y_true, y_scores, alpha=20.0)
        
        # Additional metrics
        auprc = average_precision_score(y_true, y_scores)
        
        return {
            'AUROC': auroc,
            'AUPRC': auprc,
            'EF_1%': ef_1,
            'EF_5%': ef_5,
            'BEDROC': bedroc,
            'n_actives': n_actives,
            'n_decoys': n_decoys,
            'n_total': len(y_true)
        }, "Success"
    except Exception as e:
        return None, f"Error calculating metrics: {e}"

def create_roc_curve_plot(filtered_results: List[Dict], uniprot_id: str, output_dir: str):
    """Create ROC curve plot for a target"""
    if not filtered_results or len(set([r['active_or_decoy'] for r in filtered_results])) < 2:
        return
    
    y_true = [r['active_or_decoy'] for r in filtered_results]
    y_scores = [r['score'] for r in filtered_results]
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = roc_auc_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {uniprot_id}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f'{uniprot_id}_roc_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_enrichment_plot(filtered_results: List[Dict], uniprot_id: str, output_dir: str):
    """Create enrichment plot for a target"""
    if not filtered_results:
        return
    
    # Sort by score (descending)
    sorted_results = sorted(filtered_results, key=lambda x: x['score'], reverse=True)
    y_true = [r['active_or_decoy'] for r in sorted_results]
    
    n_total = len(y_true)
    n_actives = sum(y_true)
    
    # Calculate cumulative enrichment
    percentages = np.arange(1, 101)  # 1% to 100%
    enrichments = []
    
    for pct in percentages:
        top_n = max(1, int(pct/100 * n_total))
        top_actives = sum(y_true[:top_n])
        expected_actives = (pct/100) * n_actives
        enrichment = top_actives / expected_actives if expected_actives > 0 else 1
        enrichments.append(enrichment)
    
    plt.figure(figsize=(10, 6))
    plt.plot(percentages, enrichments, linewidth=2, label=f'{uniprot_id}')
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Random')
    plt.xlabel('Percentage of Database Screened (%)')
    plt.ylabel('Enrichment Factor')
    plt.title(f'Enrichment Plot - {uniprot_id}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 100])
    
    plot_path = os.path.join(output_dir, f'{uniprot_id}_enrichment.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def analyze_single_target(results_dir: str, uniprot_id: str, output_dir: str, 
                         data_dir: str = '/home/j2ho/projects/vs_benchmarks/data') -> Optional[Dict]:
    """Analyze results for a single target using real actives/decoys"""
    print(f"\nProcessing target: {uniprot_id}")
    
    # Check for decoy file (same as reference script)
    decoy_file = os.path.join(data_dir, uniprot_id, 'decoy_per_active_d3.csv')
    if not os.path.exists(decoy_file):
        print(f"  No decoy file found: {decoy_file}")
        return None
    
    # Read decoy relationships
    actives, decoys, active_to_decoys = read_decoy_data(decoy_file)
    print(f"  Dataset: {len(actives)} actives, {len(decoys)} decoys")
    
    # Load MotifScreen results
    target_dir = os.path.join(results_dir, uniprot_id)
    results_df = load_target_results(target_dir, uniprot_id)
    
    if results_df is None or len(results_df) == 0:
        print(f"  No MotifScreen results found for {uniprot_id}")
        return None
    
    print(f"  Loaded {len(results_df)} compounds from MotifScreen results")
    
    # Apply enhanced exclusion logic (same as reference script)
    filtered_results, actives_count, decoys_count, excluded_actives, excluded_decoys = apply_enhanced_exclusion(
        results_df, actives, decoys, active_to_decoys
    )
    
    print(f"  With scores: {actives_count} actives, {decoys_count} decoys")
    print(f"  Excluded: {excluded_actives} actives, {excluded_decoys} decoys")
    
    # Calculate metrics
    metrics, status_msg = calculate_metrics(filtered_results)
    
    # Create target-specific output directory
    target_output_dir = os.path.join(output_dir, uniprot_id)
    os.makedirs(target_output_dir, exist_ok=True)
    
    # Save detailed results
    if filtered_results:
        df_results = pd.DataFrame(filtered_results)
        df_results.to_csv(os.path.join(target_output_dir, f'{uniprot_id}_detailed_results.csv'), index=False)
    
    # Track statistics
    target_stat = {
        'target': uniprot_id,
        'method': 'motifscreen',
        'total_actives': len(actives),
        'total_decoys': len(decoys),
        'actives_with_score': actives_count,
        'decoys_with_score': decoys_count,
        'excluded_actives': excluded_actives,
        'excluded_decoys': excluded_decoys,
        'metrics_calculated': metrics is not None,
        'reason': status_msg
    }
    
    if metrics:
        print(f"  AUROC: {metrics['AUROC']:.3f}")
        print(f"  EF 1%: {metrics['EF_1%']:.2f}")
        print(f"  EF 5%: {metrics['EF_5%']:.2f}")
        print(f"  BEDROC: {metrics['BEDROC']:.3f}")
        
        metrics.update({
            'target': uniprot_id,
            'method': 'motifscreen'
        })
        
        # Create plots
        roc_plot = create_roc_curve_plot(filtered_results, uniprot_id, target_output_dir)
        enrichment_plot = create_enrichment_plot(filtered_results, uniprot_id, target_output_dir)
        
        if roc_plot:
            metrics['roc_plot'] = roc_plot
        if enrichment_plot:
            metrics['enrichment_plot'] = enrichment_plot
    else:
        print(f"  Metrics not calculated: {status_msg}")
    
    # Save metrics and statistics
    with open(os.path.join(target_output_dir, f'{uniprot_id}_metrics.json'), 'w') as f:
        result_data = {
            'metrics': metrics,
            'statistics': target_stat
        }
        # Convert numpy types to Python types for JSON serialization
        result_data = json.loads(json.dumps(result_data, default=lambda x: float(x) if isinstance(x, (np.integer, np.floating)) else str(x)))
        json.dump(result_data, f, indent=2)
    
    return metrics, target_stat

def create_summary_plots(all_metrics: List[Dict], output_dir: str, model_name: str):
    """Create summary plots across all targets"""
    metrics_df = pd.DataFrame(all_metrics)
    
    # AUROC distribution
    plt.figure(figsize=(10, 6))
    valid_aurocs = metrics_df['AUROC'].dropna()
    plt.hist(valid_aurocs, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(valid_aurocs.median(), color='red', linestyle='--', 
                label=f'Median: {valid_aurocs.median():.3f}')
    plt.axvline(valid_aurocs.mean(), color='orange', linestyle='--', 
                label=f'Mean: {valid_aurocs.mean():.3f}')
    plt.xlabel('AUROC')
    plt.ylabel('Number of Targets')
    plt.title(f'AUROC Distribution - {model_name}\n({len(valid_aurocs)} targets)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{model_name}_auroc_distribution.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance summary table
    summary_stats = {
        'metric': ['AUROC', 'AUPRC', 'EF@1%', 'EF@5%', 'BEDROC'],
        'mean': [
            metrics_df['AUROC'].mean(),
            metrics_df['AUPRC'].mean(),
            metrics_df['EF_1%'].mean(),
            metrics_df['EF_5%'].mean(),
            metrics_df['BEDROC'].mean()
        ],
        'median': [
            metrics_df['AUROC'].median(),
            metrics_df['AUPRC'].median(),
            metrics_df['EF_1%'].median(),
            metrics_df['EF_5%'].median(),
            metrics_df['BEDROC'].median()
        ],
        'std': [
            metrics_df['AUROC'].std(),
            metrics_df['AUPRC'].std(),
            metrics_df['EF_1%'].std(),
            metrics_df['EF_5%'].std(),
            metrics_df['BEDROC'].std()
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(output_dir, f'{model_name}_summary_stats.csv'), index=False)
    
    return summary_df

def main():
    parser = argparse.ArgumentParser(description='Analyze MotifScreen benchmark results and calculate AUROC per target')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing per-target results')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name for output files')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                       help='Output directory for analysis results')
    parser.add_argument('--data_dir', type=str, default='/home/j2ho/projects/vs_benchmarks/data',
                       help='Directory containing decoy files')
    parser.add_argument('--targets', type=str, nargs='+',
                       help='Specific targets to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Create output directory
    analysis_output = os.path.join(args.output_dir, f"{args.model_name}_analysis")
    os.makedirs(analysis_output, exist_ok=True)
    
    # Find all target directories
    if args.targets:
        target_dirs = [t for t in args.targets if os.path.exists(os.path.join(args.results_dir, t))]
    else:
        target_dirs = [d for d in os.listdir(args.results_dir) 
                      if os.path.isdir(os.path.join(args.results_dir, d))]
    
    if not target_dirs:
        print("No target directories found!")
        return
    
    print(f"Found {len(target_dirs)} targets to analyze")
    print("="*60)
    
    # Analyze each target
    all_metrics = []
    all_target_stats = []
    failed_targets = []
    
    for uniprot_id in sorted(target_dirs):
        try:
            result = analyze_single_target(
                args.results_dir, uniprot_id, analysis_output, args.data_dir)
            
            if result:
                metrics, target_stat = result
                if metrics:
                    all_metrics.append(metrics)
                all_target_stats.append(target_stat)
            else:
                failed_targets.append(uniprot_id)
        except Exception as e:
            print(f"Error analyzing {uniprot_id}: {e}")
            failed_targets.append(uniprot_id)
        print("-" * 40)
    
    if not all_metrics:
        print("No successful analyses!")
        return
    
    # Save combined results
    metrics_df = pd.DataFrame(all_metrics)
    stats_df = pd.DataFrame(all_target_stats)
    
    metrics_df.to_csv(os.path.join(analysis_output, f'{args.model_name}_all_metrics.csv'), index=False)
    stats_df.to_csv(os.path.join(analysis_output, f'{args.model_name}_target_statistics.csv'), index=False)
    
    # Create summary plots
    summary_stats = create_summary_plots(all_metrics, analysis_output, args.model_name)
    
    # Final summary (same format as reference script)
    print("="*60)
    print(f"SUMMARY - {args.model_name.upper()}")
    print("="*60)
    print(f"Targets with metrics: {len(all_metrics)}")
    if all_metrics:
        print(f"Mean AUROC: {metrics_df['AUROC'].mean():.3f} ± {metrics_df['AUROC'].std():.3f}")
        print(f"Mean EF 1%: {metrics_df['EF_1%'].mean():.2f} ± {metrics_df['EF_1%'].std():.2f}")
        print(f"Mean EF 5%: {metrics_df['EF_5%'].mean():.2f} ± {metrics_df['EF_5%'].std():.2f}")
        print(f"Mean BEDROC: {metrics_df['BEDROC'].mean():.3f} ± {metrics_df['BEDROC'].std():.3f}")
    
    print(f"Failed analyses: {len(failed_targets)}")
    if failed_targets:
        print(f"Failed targets: {', '.join(failed_targets[:10])}{'...' if len(failed_targets) > 10 else ''}")
    
    print(f"\nResults saved to: {analysis_output}")
    print("="*60)

if __name__ == "__main__":
    main()