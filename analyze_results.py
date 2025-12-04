#!/usr/bin/env python3
"""
Comprehensive probe evaluation analysis with beautiful visualizations
"""

import pandas as pd
import numpy as np
import json
import glob
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
warnings.filterwarnings('ignore')

# Set style for beautiful plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def read_eval_probe_output(results_dir="results"):
    """
    Read evaluation results from JSON files
    Handles eval_probe.py JSON format with datasets array
    Supports filenames: {model}_{dataset}_{probe}_eval.json
    """
    results_path = Path(results_dir)
    json_files = list(results_path.glob("*_eval.json"))
    
    if not json_files:
        print(f"No JSON files found in {results_dir}/")
        return None
    
    data = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                result = json.load(f)
            
            # Extract probe type from filename: {model}_{dataset}_{probe}_eval.json
            filename = json_file.stem  # Remove .json
            parts = filename.split('_')
            
            # Handle different filename patterns
            if len(parts) >= 4 and parts[-1] == 'eval':
                # Pattern: model_dataset_probe_eval.json
                probe_key = parts[-2]
            else:
                # Fallback: assume mlp if not specified
                probe_key = 'mlp'
            
            # Map probe names
            probe_map = {
                "mlp": "MLP",
                "logreg": "LogReg",
                "logreg_cal": "LogReg+Cal",
                "tree": "Tree",
                "xform": "Transformer"
            }
            probe = probe_map.get(probe_key, probe_key)
            
            # Extract model name from JSON
            model_key = result.get('model', '')
            
            # Map to readable names
            model_map = {
                "llama": "Llama-3.1-8B",
                "llama31": "Llama-3.1-8B",  # Support both
                "qwen": "Qwen2.5-7B"
            }
            model = model_map.get(model_key, model_key)
            
            # The JSON has a 'datasets' array with one entry per dataset
            datasets_list = result.get('datasets', [])
            
            for dataset_result in datasets_list:
                dataset_key = dataset_result.get('dataset', '')
                
                # Map dataset names
                dataset_map = {
                    "triviaqa": "TriviaQA",
                    "hotpotqa": "HotpotQA",
                    "squad_v2": "SQuADv2",
                    "gsm8k": "GSM8K",
                    "mmlu": "MMLU"
                }
                dataset = dataset_map.get(dataset_key, dataset_key)
                
                # Extract metrics
                data.append({
                    'Model': model,
                    'Probe': probe,
                    'Dataset': dataset,
                    'N': dataset_result.get('n_examples', 0),
                    'Model_Acc': dataset_result.get('model_accuracy', 0.0),
                    'Probe_Acc': dataset_result.get('probe_accuracy', 0.0),
                    'Precision': dataset_result.get('probe_precision', 0.0),
                    'Recall': dataset_result.get('probe_recall', 0.0),
                    'F1': dataset_result.get('probe_f1', 0.0),
                    'Threshold': dataset_result.get('threshold', 0.0),
                    'TP': dataset_result.get('tp', 0),
                    'FP': dataset_result.get('fp', 0),
                    'TN': dataset_result.get('tn', 0),
                    'FN': dataset_result.get('fn', 0),
                })
            
            print(f"✓ Loaded: {model} - {dataset} - {probe}")
            
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    df = df.sort_values(['Model', 'Probe', 'Dataset']).reset_index(drop=True)
    return df

def create_beautiful_plots(df, output_dir="results/figures"):
    """Create publication-quality visualizations"""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Color scheme
    colors = {
        'Llama-3.1-8B': '#FF6B6B',  # Coral red
        'Qwen2.5-7B': '#4ECDC4',    # Turquoise
    }
    
    # 1. Probe Type Comparison - Grouped Bar Chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    metrics = ['Probe_Acc', 'Precision', 'Recall', 'F1']
    metric_labels = ['Probe Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    # Group by Model and Probe, average across datasets
    grouped = df.groupby(['Model', 'Probe'])[metrics].mean().reset_index()
    
    probes = grouped['Probe'].unique()
    models = grouped['Model'].unique()
    n_probes = len(probes)
    n_models = len(models)
    
    x = np.arange(len(metrics))
    width = 0.08  # Narrower bars for more groups
    
    # Different colors for each probe type
    probe_colors = {
        'MLP': '#E74C3C',
        'LogReg': '#3498DB', 
        'LogReg+Cal': '#9B59B6',
        'Tree': '#2ECC71',
        'Transformer': '#F39C12'
    }
    
    idx = 0
    for model in models:
        for probe in probes:
            subset = grouped[(grouped['Model'] == model) & (grouped['Probe'] == probe)]
            if subset.empty:
                continue
            
            offset = (idx - (n_probes * n_models) / 2) * width
            bars = ax.bar(x + offset, subset[metrics].values[0], width,
                         label=f'{model} - {probe}',
                         color=probe_colors.get(probe, '#95A5A6'),
                         alpha=0.8 if model == models[0] else 0.6,
                         edgecolor='white', linewidth=1.5)
            
            # Add value labels on bars (only for top 3 performers)
            if idx < 3:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=8)
            idx += 1
    
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Probe Architecture Comparison Across All Datasets\n(Averaged Performance)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', frameon=True, shadow=True, ncol=2, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/model_comparison.png")
    plt.close()
    
    # 2. Per-Dataset Heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, model in enumerate(df['Model'].unique()):
        model_df = df[df['Model'] == model]
        heatmap_data = model_df.pivot_table(
            index='Dataset', 
            values=['Probe_Acc', 'Precision', 'Recall', 'F1']
        )
        heatmap_data.columns = ['F1', 'Precision', 'Probe Acc', 'Recall']
        heatmap_data = heatmap_data[['Probe Acc', 'Precision', 'Recall', 'F1']]
        
        ax = ax1 if idx == 0 else ax2
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Score'},
                   linewidths=2, linecolor='white')
        ax.set_title(f'{model}\nPer-Dataset Performance', 
                    fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('Dataset', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dataset_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/dataset_heatmap.png")
    plt.close()
    
    # 2b. NEW: Probe Type Performance by Dataset
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    datasets = df['Dataset'].unique()
    probes = df['Probe'].unique()
    
    for idx, dataset in enumerate(datasets):
        if idx >= 6:
            break
        ax = axes[idx]
        
        dataset_data = df[df['Dataset'] == dataset].groupby(['Model', 'Probe'])['F1'].mean().reset_index()
        
        # Pivot for grouped bar chart
        pivot = dataset_data.pivot(index='Probe', columns='Model', values='F1')
        
        x = np.arange(len(pivot.index))
        width = 0.35
        
        colors_local = {'Llama-3.1-8B': '#FF6B6B', 'Qwen2.5-7B': '#4ECDC4'}
        
        for i, model in enumerate(pivot.columns):
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, pivot[model], width, 
                         label=model, color=colors_local[model], alpha=0.8)
            
            # Add values
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'{dataset}', fontsize=13, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=45, ha='right')
        ax.set_ylim(0, 1.0)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Hide unused subplot
    if len(datasets) < 6:
        for idx in range(len(datasets), 6):
            axes[idx].set_visible(False)
    
    plt.suptitle('Probe Performance by Dataset\n(F1 Score Comparison)', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/probe_by_dataset.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/probe_by_dataset.png")
    plt.close()
    
    # 3. Precision vs Recall Scatter
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        
        # Plot points
        scatter = ax.scatter(model_df['Recall'], model_df['Precision'],
                           s=300, alpha=0.7, color=colors[model],
                           edgecolors='black', linewidth=2, label=model)
        
        # Add dataset labels
        for _, row in model_df.iterrows():
            ax.annotate(row['Dataset'], 
                       (row['Recall'], row['Precision']),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=9, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', 
                                facecolor=colors[model], alpha=0.3))
    
    # Add diagonal line (F1 contours)
    recall_range = np.linspace(0.01, 1, 100)
    for f1 in [0.3, 0.5, 0.7, 0.9]:
        precision = f1 * recall_range / (2 * recall_range - f1)
        precision = np.clip(precision, 0, 1)
        ax.plot(recall_range, precision, '--', alpha=0.3, color='gray', linewidth=1)
        ax.text(0.95, f1 * 0.95 / (2 * 0.95 - f1), f'F1={f1}', 
               fontsize=9, color='gray', ha='right')
    
    ax.set_xlabel('Recall', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision vs Recall Trade-off\n(Dashed lines show F1 score contours)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', frameon=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/precision_recall.png")
    plt.close()
    
    # 4. Confusion Matrix Visualization (Best probe per model-dataset)
    # Select only MLP probe for clarity (best performer)
    df_mlp = df[df['Probe'] == 'MLP'].copy()
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(df_mlp.iterrows()):
        if idx >= 10:
            break
        ax = axes[idx]
        
        # Create confusion matrix
        cm = np.array([[row['TP'], row['FN']], 
                      [row['FP'], row['TN']]])
        
        # Normalize to percentages
        cm_pct = cm / cm.sum() * 100
        
        # Plot
        sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax,
                   cbar=False, square=True,
                   xticklabels=['Pred Wrong', 'Pred Correct'],  # FIXED: Swapped order
                   yticklabels=['True Correct', 'True Wrong'],
                   linewidths=2, linecolor='white')
        
        ax.set_title(f'{row["Model"]}\n{row["Dataset"]}\n(P={row["Precision"]:.2f}, R={row["Recall"]:.2f})',
                    fontsize=10, fontweight='bold')
        
    # Hide unused subplots
    for idx in range(len(df_mlp), 10):
        axes[idx].set_visible(False)
    
    plt.suptitle('Confusion Matrices - MLP Probe (% of Total Examples)', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/confusion_matrices.png")
    plt.close()
    
    # 5. Model Accuracy vs Probe Performance
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        
        # Plot Model Acc vs F1
        ax.scatter(model_df['Model_Acc'] * 100, model_df['F1'],
                  s=400, alpha=0.7, color=colors[model],
                  edgecolors='black', linewidth=2, label=model)
        
        # Add dataset labels
        for _, row in model_df.iterrows():
            ax.annotate(row['Dataset'],
                       (row['Model_Acc'] * 100, row['F1']),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5',
                                facecolor=colors[model], alpha=0.3))
    
    ax.set_xlabel('Model Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Probe F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Probe Performance vs Model Accuracy\n(Lower model accuracy → Easier for probe to predict "wrong")',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='best', frameon=True, shadow=True, fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_acc_vs_probe.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/model_acc_vs_probe.png")
    plt.close()
    
    # 6. False Positive Rate Analysis - By Probe Type
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Calculate FP rate (handle division by zero)
    df_plot = df.copy()
    df_plot['Total_Wrong'] = df_plot['FP'] + df_plot['TN']
    df_plot['FP_Rate'] = df_plot.apply(
        lambda row: row['FP'] / row['Total_Wrong'] if row['Total_Wrong'] > 0 else 0, 
        axis=1
    )
    df_plot['FP_Rate_pct'] = df_plot['FP_Rate'] * 100
    
    # Create a combined identifier for grouping
    df_plot['Model_Probe'] = df_plot['Model'] + ' - ' + df_plot['Probe']
    
    # Get unique datasets and model-probe combinations
    datasets = sorted(df_plot['Dataset'].unique())
    model_probes = sorted(df_plot['Model_Probe'].unique())
    
    x_pos = np.arange(len(datasets))
    n_bars = len(model_probes)
    width = 0.8 / n_bars  # Adjust width based on number of bars
    
    # Create color map for probe types
    probe_colors = {
        'MLP': '#2E86AB',
        'LogReg': '#A23B72', 
        'cal': '#F18F01',
        'Tree': '#C73E1D',
        'Transformer': '#6A994E'
    }
    
    for idx, model_probe in enumerate(model_probes):
        data = df_plot[df_plot['Model_Probe'] == model_probe].set_index('Dataset')
        # Reindex to match datasets order (fills missing with 0)
        data = data.reindex(datasets, fill_value=0)
        offset = (idx - n_bars/2 + 0.5) * width
        
        # Extract probe type for color
        probe_type = model_probe.split(' - ')[1]
        color = probe_colors.get(probe_type, '#999999')
        
        bars = ax.bar(x_pos + offset, 
                     data['FP_Rate_pct'].values,
                     width, label=model_probe, color=color,
                     alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels for significant values only (> 5%)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if height > 5:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}',
                       ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('False Positive Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('False Positive Rate by Dataset and Probe Type\n(Lower is better - probe correctly identifies wrong answers)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.legend(loc='upper right', frameon=True, shadow=True, ncol=2, fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, min(df_plot['FP_Rate_pct'].max() * 1.15, 100))  # Cap at 100%
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/false_positive_rate.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/false_positive_rate.png")
    plt.close()
    
    print(f"\n✓ All visualizations saved to {output_dir}/")
    print(f"    Total files: 7 visualization PNGs")
    
    # Print probe ranking
    print("\n" + "="*60)
    print("PROBE RANKING (by average F1 across all datasets)")
    print("="*60)
    probe_ranking = df.groupby(['Model', 'Probe'])['F1'].mean().reset_index()
    probe_ranking = probe_ranking.sort_values(['Model', 'F1'], ascending=[True, False])
    for model in probe_ranking['Model'].unique():
        print(f"\n{model}:")
        model_probes = probe_ranking[probe_ranking['Model'] == model]
        for idx, row in enumerate(model_probes.itertuples(), 1):
            print(f"  {idx}. {row.Probe:15s} F1={row.F1:.3f}")


def main():
    print("="*80)
    print("COMPREHENSIVE PROBE EVALUATION ANALYSIS")
    print("="*80)
    print()
    
    # Try to read JSON files
    df = read_eval_probe_output("results")
    
    if df is None or df.empty:
        print("\n" + "="*80)
        print("ERROR: Could not read evaluation results")
        print("="*80)
        print("\nPlease ensure JSON files exist in results/ directory")
        print("Expected files: llama31_triviaqa_eval.json, etc.")
        return
    
    print(f"\n✓ Successfully loaded {len(df)} evaluations")
    print()
    
    # Print summary table
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print()
    print(df[['Model', 'Probe', 'Dataset', 'Model_Acc', 'Probe_Acc', 'Precision', 'Recall', 'F1']].to_string(index=False))
    print()
    
    # Create visualizations
    print("="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    print()
    
    create_beautiful_plots(df)
    
    # Save CSV
    df.to_csv('results/probe_evaluation_summary.csv', index=False)
    print(f"\n✓ Summary CSV saved to: results/probe_evaluation_summary.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print()
    print("Generated files:")
    print("  - results/figures/*.png (7 visualization files)")
    print("  - results/probe_evaluation_summary.csv")
    print()
    print("Visualizations:")
    print("  1. model_comparison.png - Probe architecture comparison")
    print("  2. dataset_heatmap.png - Per-dataset performance heatmaps")
    print("  3. probe_by_dataset.png - Probe F1 scores by dataset")
    print("  4. precision_recall.png - Precision vs Recall scatter")
    print("  5. confusion_matrices.png - Confusion matrices (MLP only)")
    print("  6. model_acc_vs_probe.png - Model accuracy vs probe F1")
    print("  7. false_positive_rate.png - FPR by dataset")
    print()

if __name__ == "__main__":
    main()