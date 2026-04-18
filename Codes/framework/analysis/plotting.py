import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def create_comprehensive_plots(results_df: pd.DataFrame, plots_dir: str):
    print("\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS...")
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle('SVD Compression Analysis: Comprehensive Results', fontsize=16, fontweight='bold')
    
    # Accuracy vs Rank (by model)
    ax1 = axes[0, 0]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax1.plot(model_data['rank'], model_data['compressed_accuracy'], marker='o', label=model, linewidth=2)
    ax1.set_xlabel('SVD Rank'); ax1.set_ylabel('Accuracy'); ax1.set_title('Accuracy vs Rank (by Model)')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # Accuracy vs Rank (by dataset)
    ax2 = axes[0, 1]
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        ax2.plot(dataset_data['rank'], dataset_data['compressed_accuracy'], marker='s', label=dataset, linewidth=2)
    ax2.set_xlabel('SVD Rank'); ax2.set_ylabel('Accuracy'); ax2.set_title('Accuracy vs Rank (by Dataset)')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    # Speedup factor
    if 'speedup_factor' in results_df.columns:
        ax3 = axes[0, 2]
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            speedup_by_rank = model_data.groupby('rank')['speedup_factor'].mean()
            ax3.plot(speedup_by_rank.index, speedup_by_rank.values, marker='^', label=model, linewidth=2)
        ax3.set_xlabel('SVD Rank'); ax3.set_ylabel('Speedup Factor'); ax3.set_title('Inference Speedup vs Rank')
        ax3.legend(); ax3.grid(True, alpha=0.3)
    
    # Parameter reduction (bar)
    ax4 = axes[1, 0]
    if 'param_reduction' in results_df.columns:
        param_data = results_df.groupby(['model', 'rank'])['param_reduction'].mean().reset_index()
        for model in param_data['model'].unique():
            model_param_data = param_data[param_data['model'] == model]
            ax4.bar(model_param_data['rank'] + (ord(model[0]) - 97) * 0.2, model_param_data['param_reduction'], width=0.2, label=model)
        ax4.set_xlabel('SVD Rank'); ax4.set_ylabel('Parameters Reduced'); ax4.set_title('Parameter Reduction by Model')
        ax4.legend(); ax4.grid(True, alpha=0.3, axis='y')
    
    # Accuracy drop distribution
    if 'accuracy_drop' in results_df.columns:
        ax5 = axes[1, 1]
        accuracy_drops = results_df['accuracy_drop']
        ax5.hist(accuracy_drops, bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
        ax5.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax5.set_xlabel('Accuracy Drop'); ax5.set_ylabel('Frequency'); ax5.set_title('Distribution of Accuracy Drops')
        ax5.grid(True, alpha=0.3)
    
    # Memory reduction
    if 'memory_reduction' in results_df.columns:
        ax6 = axes[1, 2]
        memory_data = results_df.groupby('rank')['memory_reduction'].mean().reset_index()
        ax6.bar(memory_data['rank'], memory_data['memory_reduction'], color='lightgreen', edgecolor='darkgreen', linewidth=2)
        ax6.set_xlabel('SVD Rank'); ax6.set_ylabel('Memory Reduction (MB)'); ax6.set_title('Average Memory Reduction')
        ax6.grid(True, alpha=0.3, axis='y')
    
    # Accuracy by task type
    if 'task_type' in results_df.columns:
        ax7 = axes[2, 0]
        task_data = results_df.groupby(['task_type', 'rank'])['compressed_accuracy'].mean().reset_index()
        for task_type in task_data['task_type'].unique():
            task_type_data = task_data[task_data['task_type'] == task_type]
            ax7.plot(task_type_data['rank'], task_type_data['compressed_accuracy'], marker='D', label=task_type, linewidth=2)
        ax7.set_xlabel('SVD Rank'); ax7.set_ylabel('Accuracy'); ax7.set_title('Accuracy by Task Type')
        ax7.legend(); ax7.grid(True, alpha=0.3)
    
    # Compression ratio heatmap
    if 'compression_ratio' in results_df.columns:
        ax8 = axes[2, 1]
        pivot_data = results_df.pivot_table(index='model', columns='rank', values='compression_ratio', aggfunc='mean')
        im = ax8.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
        ax8.set_xticks(range(len(pivot_data.columns))); ax8.set_xticklabels(pivot_data.columns)
        ax8.set_yticks(range(len(pivot_data.index))); ax8.set_yticklabels(pivot_data.index)
        ax8.set_xlabel('SVD Rank'); ax8.set_ylabel('Model'); ax8.set_title('Compression Ratio Heatmap')
        plt.colorbar(im, ax=ax8, label='Compression Ratio')
    
    # Best config per dataset
    ax9 = axes[2, 2]
    best_configs = []
    for dataset in results_df['dataset'].unique():
        dataset_data = results_df[results_df['dataset'] == dataset]
        best_idx = dataset_data['compressed_accuracy'].idxmax()
        best_configs.append(dataset_data.loc[best_idx])
    if best_configs:
        best_df = pd.DataFrame(best_configs)
        y_pos = np.arange(len(best_df))
        ax9.barh(y_pos, best_df['compressed_accuracy'], color='skyblue', edgecolor='navy')
        ax9.set_yticks(y_pos); ax9.set_yticklabels(best_df['dataset'])
        ax9.set_xlabel('Accuracy'); ax9.set_title('Best Configuration for Each Dataset')
        for i, (acc, model, rank) in enumerate(zip(best_df['compressed_accuracy'], best_df['model'], best_df['rank'])):
            ax9.text(acc + 0.01, i, f'{model} (R{rank})', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(plots_dir) / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()