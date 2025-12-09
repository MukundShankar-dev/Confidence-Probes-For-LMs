#!/usr/bin/env python3
"""
Analyze probe transfer results across model sizes

Generates visualizations and reports showing:
1. F1 comparison across models
2. Transfer degradation analysis
3. Per-dataset transfer patterns
4. Model size vs performance
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_results():
    """Load all transfer experiment results"""
    results_dir = Path('results/transfer')
    
    # Map model names to filename patterns
    # Files are named: {model_name}_{dataset}_{probe}_eval.json
    # NOTE: Qwen3-8B excluded - it's a base model, not instruct-tuned
    models_to_test = {
        'Qwen2.5-1.5B': 'qwen2.5_1.5b',
        'Qwen3-4B': 'qwen3_4b',
    }
    
    # Also load baseline if available
    baseline_files = list(Path('results').glob('qwen_*_mlp_eval.json'))
    
    data = []
    
    # Load baseline (Qwen2.5-7B) from original results
    for baseline_file in baseline_files:
        try:
            with open(baseline_file) as f:
                result = json.load(f)
            
            for dataset_result in result.get('datasets', []):
                dataset = dataset_result['dataset']
                
                # Get all threshold results
                thresh_results = dataset_result.get('threshold_results', [])
                
                for thresh_result in thresh_results:
                    data.append({
                        'Model': 'Qwen2.5-7B',
                        'Dataset': dataset.upper(),
                        'Model_Size': '7B',
                        'Model_Family': 'Qwen2.5',
                        'Model_Acc': dataset_result['model_accuracy'],
                        'Threshold': thresh_result['threshold'],
                        'F1': thresh_result['f1'],
                        'Precision': thresh_result['precision'],
                        'Recall': thresh_result['recall'],
                        'Accuracy': thresh_result['accuracy'],
                    })
        except Exception as e:
            print(f"Warning: Could not load baseline {baseline_file}: {e}")
    
    # Load transfer results
    for model_name, file_prefix in models_to_test.items():
        # Find all files for this model (any probe type)
        model_files = list(results_dir.glob(f'{file_prefix}_*_eval.json'))
        
        for filepath in model_files:
            try:
                with open(filepath) as f:
                    result = json.load(f)
                
                for dataset_result in result.get('datasets', []):
                    dataset = dataset_result['dataset']
                    
                    # Get all threshold results
                    thresh_results = dataset_result.get('threshold_results', [])
                    
                    for thresh_result in thresh_results:
                        data.append({
                            'Model': model_name,
                            'Dataset': dataset.upper(),
                            'Model_Size': model_name.split('-')[-1].replace('B', ''),
                            'Model_Family': 'Qwen2.5' if 'Qwen2.5' in model_name else 'Qwen3',
                            'Model_Acc': dataset_result['model_accuracy'],
                            'Threshold': thresh_result['threshold'],
                            'F1': thresh_result['f1'],
                            'Precision': thresh_result['precision'],
                            'Recall': thresh_result['recall'],
                            'Accuracy': thresh_result['accuracy'],
                        })
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
    
    return pd.DataFrame(data)


def create_visualizations(df, output_dir='results/transfer/figures'):
    """Create comprehensive transfer analysis plots"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Filter to threshold 0.3 (best threshold)
    df_03 = df[df['Threshold'] == 0.3].copy()
    
    # 1. F1 Score Comparison Across Models
    fig, ax = plt.subplots(figsize=(14, 8))
    
    models = df_03['Model'].unique()
    datasets = sorted(df_03['Dataset'].unique())
    
    x = np.arange(len(datasets))
    width = 0.2
    
    colors = {
        'Qwen2.5-7B': '#E74C3C',    # Red (baseline)
        'Qwen2.5-1.5B': '#F39C12',  # Orange
        'Qwen3-4B': '#3498DB',      # Blue
        'Qwen3-8B': '#2ECC71',      # Green
    }
    
    for idx, model in enumerate(sorted(models)):
        model_data = df_03[df_03['Model'] == model].set_index('Dataset')
        model_data = model_data.reindex(datasets, fill_value=0)
        offset = (idx - len(models)/2 + 0.5) * width
        
        bars = ax.bar(x + offset, model_data['F1'].values, width,
                     label=model, color=colors.get(model, '#999999'),
                     alpha=0.8, edgecolor='white', linewidth=2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Probe Transfer Performance Across Model Sizes (Threshold=0.3)',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transfer_f1_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/transfer_f1_comparison.png")
    plt.close()
    
    # 2. Average F1 by Model
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_f1 = df_03.groupby('Model')['F1'].mean().sort_values(ascending=True)
    
    bars = ax.barh(range(len(avg_f1)), avg_f1.values,
                   color=[colors.get(m, '#999999') for m in avg_f1.index],
                   alpha=0.8, edgecolor='white', linewidth=2)
    
    # Add value labels
    for i, (model, f1) in enumerate(avg_f1.items()):
        ax.text(f1, i, f' {f1:.3f}', va='center', fontsize=12, fontweight='bold')
    
    ax.set_yticks(range(len(avg_f1)))
    ax.set_yticklabels(avg_f1.index, fontsize=12)
    ax.set_xlabel('Average F1 Score', fontsize=14, fontweight='bold')
    ax.set_title('Average F1 Across All Datasets by Model',
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    
    # Add baseline annotation
    baseline_f1 = avg_f1['Qwen2.5-7B']
    ax.axvline(baseline_f1, color='red', linestyle='--', linewidth=2, alpha=0.5,
              label=f'Baseline (7B): {baseline_f1:.3f}')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transfer_avg_f1.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/transfer_avg_f1.png")
    plt.close()
    
    # 3. Transfer Degradation Heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute degradation relative to baseline
    baseline_scores = df_03[df_03['Model'] == 'Qwen2.5-7B'].set_index('Dataset')['F1']
    
    degradation_data = []
    for model in sorted(models):
        if model == 'Qwen2.5-7B':
            continue
        model_scores = df_03[df_03['Model'] == model].set_index('Dataset')['F1']
        degradation = ((model_scores - baseline_scores) / baseline_scores * 100).reindex(datasets)
        degradation_data.append(degradation.values)
    
    transfer_models = [m for m in sorted(models) if m != 'Qwen2.5-7B']
    
    im = ax.imshow(np.array(degradation_data), cmap='RdYlGn', aspect='auto',
                   vmin=-50, vmax=10)
    
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(transfer_models)))
    ax.set_xticklabels(datasets)
    ax.set_yticklabels(transfer_models)
    
    # Annotate cells
    for i in range(len(transfer_models)):
        for j in range(len(datasets)):
            val = degradation_data[i][j]
            if not np.isnan(val):
                color = 'white' if val < -20 else 'black'
                text = ax.text(j, i, f'{val:.1f}%',
                             ha="center", va="center", color=color, fontsize=10)
    
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Transfer Model', fontsize=12, fontweight='bold')
    ax.set_title('Transfer Performance Change vs Baseline (Qwen2.5-7B)',
                fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax, label='% Change in F1')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transfer_degradation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/transfer_degradation_heatmap.png")
    plt.close()
    
    # 4. Model Size vs Performance
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract model sizes
    size_map = {'7B': 7.0, '1.5B': 1.5, '4B': 4.0, '8B': 8.0}
    avg_f1_by_model = df_03.groupby('Model')['F1'].mean()
    
    sizes = []
    f1s = []
    labels = []
    for model, f1 in avg_f1_by_model.items():
        size_str = model.split('-')[-1]
        if size_str in size_map:
            sizes.append(size_map[size_str])
            f1s.append(f1)
            labels.append(model)
    
    # Separate by family
    qwen25_mask = ['Qwen2.5' in l for l in labels]
    qwen3_mask = ['Qwen3' in l for l in labels]
    
    qwen25_sizes = [s for s, m in zip(sizes, qwen25_mask) if m]
    qwen25_f1s = [f for f, m in zip(f1s, qwen25_mask) if m]
    qwen25_labels = [l for l, m in zip(labels, qwen25_mask) if m]
    
    qwen3_sizes = [s for s, m in zip(sizes, qwen3_mask) if m]
    qwen3_f1s = [f for f, m in zip(f1s, qwen3_mask) if m]
    qwen3_labels = [l for l, m in zip(labels, qwen3_mask) if m]
    
    ax.scatter(qwen25_sizes, qwen25_f1s, s=200, alpha=0.7, color='#E74C3C',
              label='Qwen2.5 Family', edgecolors='white', linewidth=2)
    ax.scatter(qwen3_sizes, qwen3_f1s, s=200, alpha=0.7, color='#3498DB',
              label='Qwen3 Family', edgecolors='white', linewidth=2)
    
    # Add labels
    for s, f, l in zip(qwen25_sizes, qwen25_f1s, qwen25_labels):
        ax.annotate(l.split('-')[-1], (s, f), xytext=(5, 5),
                   textcoords='offset points', fontsize=10)
    for s, f, l in zip(qwen3_sizes, qwen3_f1s, qwen3_labels):
        ax.annotate(l.split('-')[-1], (s, f), xytext=(5, 5),
                   textcoords='offset points', fontsize=10)
    
    ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Transfer Performance vs Model Size',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transfer_size_vs_performance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/transfer_size_vs_performance.png")
    plt.close()
    
    # 5. Precision-Recall by Model
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in sorted(models):
        model_data = df_03[df_03['Model'] == model]
        ax.scatter(model_data['Recall'], model_data['Precision'],
                  s=100, alpha=0.7, label=model,
                  color=colors.get(model, '#999999'),
                  edgecolors='white', linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Trade-off by Model (Threshold=0.3)',
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transfer_precision_recall.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/transfer_precision_recall.png")
    plt.close()
    
    # 6. Model Accuracy vs Probe F1 (Correlation Analysis)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for model in sorted(models):
        model_data = df_03[df_03['Model'] == model]
        ax.scatter(model_data['Model_Acc'] * 100, model_data['F1'],
                  s=150, alpha=0.7, label=model,
                  color=colors.get(model, '#999999'),
                  edgecolors='white', linewidth=2)
    
    # Add trend line for all data
    all_acc = df_03['Model_Acc'].values * 100
    all_f1 = df_03['F1'].values
    
    # Remove any NaN values
    mask = ~(np.isnan(all_acc) | np.isnan(all_f1))
    if mask.sum() > 0:
        z = np.polyfit(all_acc[mask], all_f1[mask], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(all_acc[mask].min(), all_acc[mask].max(), 100)
        ax.plot(x_trend, p(x_trend), "r--", alpha=0.5, linewidth=2, 
               label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Calculate correlation
        correlation = np.corrcoef(all_acc[mask], all_f1[mask])[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
               transform=ax.transAxes, fontsize=12, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('Model Accuracy (%) on Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probe F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Base Model Accuracy vs Probe Transfer Performance',
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transfer_model_acc_vs_probe_f1.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/transfer_model_acc_vs_probe_f1.png")
    plt.close()
    
    # 7. Multi-Threshold Comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    thresholds = sorted(df['Threshold'].unique())
    
    # Plot 1: F1 vs Threshold for each model
    ax = axes[0]
    for model in sorted(models):
        model_data = df[df['Model'] == model].groupby('Threshold')['F1'].mean()
        ax.plot(model_data.index, model_data.values, 'o-', linewidth=2, 
               markersize=8, label=model, color=colors.get(model, '#999999'))
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Transfer Performance vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 2: Precision vs Threshold
    ax = axes[1]
    for model in sorted(models):
        model_data = df[df['Model'] == model].groupby('Threshold')['Precision'].mean()
        ax.plot(model_data.index, model_data.values, 'o-', linewidth=2,
               markersize=8, label=model, color=colors.get(model, '#999999'))
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 3: Recall vs Threshold
    ax = axes[2]
    for model in sorted(models):
        model_data = df[df['Model'] == model].groupby('Threshold')['Recall'].mean()
        ax.plot(model_data.index, model_data.values, 'o-', linewidth=2,
               markersize=8, label=model, color=colors.get(model, '#999999'))
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Recall', fontsize=12, fontweight='bold')
    ax.set_title('Recall vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 4: Threshold Degradation (% change from baseline at each threshold)
    ax = axes[3]
    baseline_model = 'Qwen2.5-7B'
    baseline_data = df[df['Model'] == baseline_model].groupby('Threshold')['F1'].mean()
    
    for model in sorted(models):
        if model == baseline_model:
            continue
        model_data = df[df['Model'] == model].groupby('Threshold')['F1'].mean()
        degradation = ((model_data - baseline_data) / baseline_data * 100)
        ax.plot(degradation.index, degradation.values, 'o-', linewidth=2,
               markersize=8, label=model, color=colors.get(model, '#999999'))
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Baseline')
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('% Change from Baseline', fontsize=12, fontweight='bold')
    ax.set_title('Transfer Degradation at Different Thresholds', fontsize=13, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/transfer_multi_threshold_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/transfer_multi_threshold_analysis.png")
    plt.close()
    
    # 8. Probe Architecture Comparison (if multiple probe types available)
    # Check if we have probe type information in filenames
    # This would require parsing filenames, so we'll add it if probe info is in data
    # For now, skip this one since it requires restructuring how we load data


def generate_report(df):
    """Generate text report of transfer results"""
    print("\n" + "="*70)
    print("PROBE TRANSFER ANALYSIS REPORT")
    print("="*70)
    
    # Filter to threshold 0.3
    df_03 = df[df['Threshold'] == 0.3].copy()
    
    # Overall statistics
    baseline_f1 = df_03[df_03['Model'] == 'Qwen2.5-7B']['F1'].mean()
    print(f"\nBaseline (Qwen2.5-7B): Average F1 = {baseline_f1:.3f}")
    print("\nTransfer Results:")
    print("-" * 70)
    
    for model in sorted(df_03['Model'].unique()):
        if model == 'Qwen2.5-7B':
            continue
        
        model_f1 = df_03[df_03['Model'] == model]['F1'].mean()
        degradation = ((model_f1 - baseline_f1) / baseline_f1) * 100
        
        status = "✓ Good" if degradation > -15 else "⚠ Moderate" if degradation > -30 else "✗ Poor"
        
        print(f"{model:20s} F1={model_f1:.3f}  "
              f"Change: {degradation:+.1f}%  {status}")
    
    # Per-dataset analysis
    print("\n" + "="*70)
    print("PER-DATASET TRANSFER PERFORMANCE")
    print("="*70)
    
    baseline_data = df_03[df_03['Model'] == 'Qwen2.5-7B'].set_index('Dataset')
    
    for dataset in sorted(df_03['Dataset'].unique()):
        print(f"\n{dataset}:")
        baseline_f1 = baseline_data.loc[dataset, 'F1']
        print(f"  Baseline: {baseline_f1:.3f}")
        
        for model in sorted(df_03['Model'].unique()):
            if model == 'Qwen2.5-7B':
                continue
            
            model_data = df_03[(df_03['Model'] == model) & (df_03['Dataset'] == dataset)]
            if len(model_data) == 0:
                continue
            
            model_f1 = model_data['F1'].values[0]
            change = ((model_f1 - baseline_f1) / baseline_f1) * 100
            
            print(f"  {model:20s} {model_f1:.3f}  ({change:+.1f}%)")
    
    print("\n" + "="*70)


def main():
    print("Loading transfer evaluation results...")
    df = load_results()
    
    if df.empty:
        print("ERROR: No results found. Please run eval_probe_transfer.sh first")
        return
    
    print(f"Loaded {len(df)} evaluation results")
    print(f"Models: {', '.join(df['Model'].unique())}")
    print(f"Datasets: {', '.join(df['Dataset'].unique())}")
    
    # Generate visualizations
    print("\nCreating visualizations...")
    create_visualizations(df)
    
    # Generate text report
    generate_report(df)
    
    # Save detailed results
    output_file = 'results/transfer/transfer_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  - results/transfer/figures/*.png (7 visualizations)")
    print("  - results/transfer/transfer_analysis.csv")


if __name__ == "__main__":
    main()