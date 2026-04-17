import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

def generate_plots(results_file_path):
    """
    Reads the latest JSON result and generates visualizations.
    """
    if not os.path.exists(results_file_path):
        print(f"❌ Results file not found: {results_file_path}")
        return

    with open(results_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract Data
    acc_orig = data.get('original_model', {}).get('metrics', {}).get('accuracy')
    acc_svd = data.get('compressed_model', {}).get('metrics', {}).get('accuracy')
    cm_orig = data.get('original_model', {}).get('metrics', {}).get('confusion_matrix')
    cm_svd = data.get('compressed_model', {}).get('metrics', {}).get('confusion_matrix')
    
    size_red_mb = data.get('comparison', {}).get('size_reduction_mb', 0)
    size_red_pct = data.get('comparison', {}).get('size_reduction_pct', 0)
    
    # Setup Plot Style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)

    # --- Plot 1: Accuracy Comparison ---
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['Original BERT', 'SVD Compressed (Rank 8)']
    accuracies = [acc_orig if acc_orig else 0, acc_svd if acc_svd else 0]
    colors = ['#3498db', '#e74c3c']
    
    bars = ax1.bar(models, accuracies, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Accuracy Score', fontsize=12)
    ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # --- Plot 2: Size Reduction ---
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ['Memory Saved (MB)', 'Reduction (%)']
    values = [size_red_mb, size_red_pct]
    colors_bar = ['#2ecc71', '#f1c40f']
    
    bars2 = ax2.bar(categories, values, color=colors_bar, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title('Compression Efficiency', fontsize=14, fontweight='bold')
    
    for bar, val in zip(bars2, values):
        height = bar.get_height()
        unit = 'MB' if 'MB' in str(bar.get_xticklabels()[0]) else '%'
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.2f} {unit}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Plot 3: Confusion Matrix (Original) ---
    ax3 = fig.add_subplot(gs[1, 0])
    if cm_orig:
        cm_arr = np.array(cm_orig)
        im = ax3.imshow(cm_arr, cmap='Blues', interpolation='nearest')
        ax3.set_title('Original Model Confusion Matrix', fontsize=12, fontweight='bold')
        
        # Add text annotations
        thresh = cm_arr.max() / 2.
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax3.text(j, i, format(cm_arr[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm_arr[i, j] > thresh else "black")
        ax3.set_xlabel('Predicted Label')
        ax3.set_ylabel('True Label')
    else:
        ax3.text(0.5, 0.5, 'No CM Data Available', ha='center', va='center')
        ax3.axis('off')

    # --- Plot 4: Confusion Matrix (Compressed) ---
    ax4 = fig.add_subplot(gs[1, 1])
    if cm_svd:
        cm_arr = np.array(cm_svd)
        im = ax4.imshow(cm_arr, cmap='Oranges', interpolation='nearest')
        ax4.set_title('SVD Compressed Model Confusion Matrix', fontsize=12, fontweight='bold')
        
        thresh = cm_arr.max() / 2.
        for i in range(cm_arr.shape[0]):
            for j in range(cm_arr.shape[1]):
                ax4.text(j, i, format(cm_arr[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm_arr[i, j] > thresh else "black")
        ax4.set_xlabel('Predicted Label')
        ax4.set_ylabel('True Label')
    else:
        ax4.text(0.5, 0.5, 'No CM Data Available', ha='center', va='center')
        ax4.axis('off')

    plt.suptitle(f"BERT SVD Evaluation Report\nRank: {data['config']['svd_rank']} | Device: {data['config']['device']}", 
                 fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save Figure
    output_dir = Path(results_file_path).parent
    plot_name = output_dir / "evaluation_report.png"
    plt.savefig(plot_name, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {plot_name}")
    plt.show()

if __name__ == "__main__":
    # Example usage if running standalone
    # Find the latest result file automatically
    results_dir = Path.cwd() / "experiments" / "results"
    if not results_dir.exists():
        print("Results directory not found.")
    else:
        files = sorted(results_dir.glob("results_*.json"))
        if files:
            latest_file = files[-1]
            generate_plots(latest_file)
        else:
            print("No result files found.")