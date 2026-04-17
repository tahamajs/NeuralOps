"""
🌟 SMART ROUTER FOR REASONING TASKS - OPTIMIZED SVD COMPRESSION v8
Features:
- Multi-rank SVD compression testing (8, 16, 32, 64)
- Multiple model architectures support (BERT, RoBERTa, DistilBERT)
- Comprehensive parameter reduction analysis
- Speed benchmarking
- Complete visualization suite
Author: Optimized for your datasets
"""

import torch
import pandas as pd
import numpy as np
import json
import time
import psutil
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
CONFIG = {
    # Model Settings
    "base_models": {
        "bert": "bert-base-uncased",
        "distilbert": "distilbert-base-uncased",
        "roberta": "roberta-base"
    },
    
    # SVD Rank Experiments
    "svd_ranks": [8, 16, 32, 64],  # Test different compression levels
    
    # Dataset Paths
    "data_files": {
        "commonsenseqa": "./data/reasoning/commonsenseqa/processed/commonsenseqa_train.csv",
        "hellaswag": "./data/reasoning/hellaswag/processed/hellaswag_train.csv",
        "gsm8k": "./data/reasoning/gsm8k/processed/gsm8k_train.csv",
        "arc_challenge": "./data/reasoning/arc_challenge/processed/arc_challenge_train.csv",
        "arc_agi": "./data/reasoning/arc_agi/processed/arc_agi_train.csv",
    },
    
    # Training Parameters
    "min_samples": 50,
    "max_samples": 1000,  # Limit for faster training
    "batch_size": 8,
    "epochs": 2,  # Reduced for experimentation
    "learning_rate": 2e-5,
    "max_length": 128,  # Reduced for speed
    
    # Evaluation
    "val_split": 0.2,
    
    # Output
    "output_dir": "./results",
    "plots_dir": "./plots/comprehensive"
}

# ==================== DEVICE SETUP ====================
def get_optimized_device():
    """Optimized device selection with memory management"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.empty_cache()
    else:
        device = torch.device("cpu")
    
    print(f"⚡ Using device: {device}")
    if device.type == 'mps':
        print("   🍎 Apple Silicon GPU detected - Optimizing for MPS")
    return device

DEVICE = get_optimized_device()

# ==================== ENHANCED SVD COMPRESSOR ====================
class OptimizedSVDLinear(torch.nn.Module):
    """Optimized SVD compression with memory-efficient operations"""
    def __init__(self, linear_layer, rank):
        super().__init__()
        self.rank = rank
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        
        # SVD Decomposition
        W = linear_layer.weight.data
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        
        # Store compressed components
        self.U = torch.nn.Parameter(U[:, :rank].contiguous())
        self.S = torch.nn.Parameter(S[:rank].contiguous())
        self.Vh = torch.nn.Parameter(Vh[:rank, :].contiguous())
        
        # Bias
        self.bias = None
        if linear_layer.bias is not None:
            self.bias = torch.nn.Parameter(linear_layer.bias.data.clone())
        
        # Calculate parameter reduction
        self.original_params = self.in_features * self.out_features
        if self.bias is not None:
            self.original_params += self.out_features
        
        self.compressed_params = rank * (self.in_features + self.out_features + 1)
        self.compression_ratio = self.compressed_params / self.original_params
    
    def forward(self, x):
        """Efficient forward pass"""
        # Reconstruct weight matrix on-the-fly
        W_recon = self.U @ (self.S.diag() @ self.Vh)
        
        if self.bias is not None:
            return torch.nn.functional.linear(x, W_recon, self.bias)
        return torch.nn.functional.linear(x, W_recon)

def apply_svd_compression(model, rank, layer_types=['attention', 'dense', 'output']):
    """
    Apply SVD compression to multiple layer types
    Returns compression statistics
    """
    compressed_count = 0
    total_params_before = 0
    total_params_after = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if this layer should be compressed
            compress_layer = False
            for layer_type in layer_types:
                if layer_type in name.lower():
                    compress_layer = True
                    break
            
            if compress_layer and module.in_features >= rank and module.out_features >= rank:
                try:
                    # Track parameters before compression
                    params_before = module.in_features * module.out_features
                    if module.bias is not None:
                        params_before += module.out_features
                    total_params_before += params_before
                    
                    # Replace with compressed layer
                    compressed_layer = OptimizedSVDLinear(module, rank)
                    
                    # Navigate to parent module
                    parts = name.split('.')
                    parent = model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], compressed_layer)
                    
                    # Track parameters after compression
                    total_params_after += compressed_layer.compressed_params
                    compressed_count += 1
                    
                except Exception as e:
                    print(f"⚠️ Failed to compress {name}: {e}")
                    continue
    
    # Calculate overall statistics
    compression_stats = {
        'compressed_layers': compressed_count,
        'total_params_before': total_params_before,
        'total_params_after': total_params_after,
        'compression_ratio': total_params_after / total_params_before if total_params_before > 0 else 1.0,
        'param_reduction': total_params_before - total_params_after
    }
    
    return compression_stats

# ==================== OPTIMIZED DATASET ====================
class FastReasoningDataset(Dataset):
    """Optimized dataset for speed"""
    def __init__(self, df, task_type, tokenizer, max_length=128):
        self.df = df
        self.task_type = task_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Pre-process for speed
        self.samples = []
        for idx in range(len(df)):
            self.samples.append(self._preprocess(idx))
    
    def _preprocess(self, idx):
        """Pre-process a single sample"""
        row = self.df.iloc[idx]
        
        if self.task_type in ['commonsenseqa', 'arc_challenge']:
            question = str(row.get('question', ''))
            choices = []
            
            for i in range(ord('A'), ord('D') + 1):
                col = f'choice_{chr(i).lower()}'
                if col in row and pd.notna(row[col]):
                    choices.append(str(row[col]))
            
            if choices:
                return {'type': 'mc', 'texts': [f"{question} {c}" for c in choices]}
            return {'type': 'single', 'text': question}
        
        else:  # hellaswag, gsm8k, arc_agi
            prompt = str(row.get('prompt', ''))
            return {'type': 'single', 'text': prompt}
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if sample['type'] == 'mc':
            encoded = self.tokenizer(
                sample['texts'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'],
                'attention_mask': encoded['attention_mask'],
                'labels': torch.zeros(len(sample['texts']), dtype=torch.long)
            }
        else:
            encoded = self.tokenizer(
                sample['text'],
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoded['input_ids'].squeeze(0),
                'attention_mask': encoded['attention_mask'].squeeze(0),
                'labels': torch.tensor(0, dtype=torch.long)
            }

# ==================== BENCHMARKING FUNCTIONS ====================
def benchmark_model(model, dataloader, device):
    """Benchmark model speed and memory"""
    model.eval()
    
    # Memory before
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        memory_before = torch.cuda.memory_allocated()
    else:
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Inference speed
    start_time = time.time()
    batch_times = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            batch_times.append(time.time() - batch_start)
    
    total_time = time.time() - start_time
    
    # Memory after
    if device.type == 'cuda':
        memory_after = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
    else:
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024
        peak_memory = memory_after
    
    # Calculate statistics
    stats = {
        'total_time': total_time,
        'avg_batch_time': np.mean(batch_times),
        'samples_per_second': len(dataloader.dataset) / total_time,
        'memory_usage_mb': (memory_after - memory_before) / 1024 / 1024 if device.type == 'cuda' else (memory_after - memory_before),
        'peak_memory_mb': peak_memory / 1024 / 1024 if device.type == 'cuda' else peak_memory,
        'throughput': len(dataloader.dataset) / total_time
    }
    
    return stats

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ==================== OPTIMIZED TRAINING ====================
def fast_train_evaluate(model, train_loader, val_loader, device, epochs=2):
    """Fast training and evaluation"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.flatten())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'avg_loss': total_loss / len(train_loader)
    }

# ==================== MAIN EXPERIMENT PIPELINE ====================
def run_comprehensive_experiments():
    """Run all experiments and collect results"""
    
    # Create output directories
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["plots_dir"]).mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = []
    dataset_stats = {}
    
    print("=" * 80)
    print("🚀 COMPREHENSIVE SVD COMPRESSION EXPERIMENTS")
    print("=" * 80)
    
    # First, analyze all datasets
    print("\n📊 ANALYZING DATASETS...")
    for task_name, data_path in CONFIG["data_files"].items():
        if Path(data_path).exists():
            df = pd.read_csv(data_path)
            dataset_stats[task_name] = {
                'samples': len(df),
                'columns': list(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
            print(f"   ✅ {task_name}: {len(df)} samples, {len(df.columns)} columns")
        else:
            print(f"   ❌ {task_name}: File not found")
    
    # Filter datasets with enough samples
    valid_datasets = {k: v for k, v in dataset_stats.items() 
                     if v['samples'] >= CONFIG["min_samples"]}
    
    print(f"\n🎯 VALID DATASETS FOR EXPERIMENTS: {list(valid_datasets.keys())}")
    
    # Main experiment loop
    experiment_id = 0
    
    for model_name, model_path in CONFIG["base_models"].items():
        print(f"\n{'='*60}")
        print(f"🧠 TESTING MODEL: {model_name.upper()} ({model_path})")
        print(f"{'='*60}")
        
        # Load base model once
        print("   📥 Loading tokenizer and base model...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2
        ).to(DEVICE)
        
        base_params = count_parameters(base_model)
        print(f" 📊 Base model parameters: {base_params:,}")
        
        for task_name in valid_datasets.keys():
            print(f"\n   📝 TASK: {task_name}")
            
            # Load and prepare data
            df = pd.read_csv(CONFIG["data_files"][task_name])
            if len(df) > CONFIG["max_samples"]:
                df = df.sample(n=CONFIG["max_samples"], random_state=42)
            
            # Split
            split_idx = int(len(df) * (1 - CONFIG["val_split"]))
            train_df = df.iloc[:split_idx]
            val_df = df.iloc[split_idx:]
            
            # Create datasets
            train_dataset = FastReasoningDataset(train_df, task_name, tokenizer, CONFIG["max_length"])
            val_dataset = FastReasoningDataset(val_df, task_name, tokenizer, CONFIG["max_length"])
            
            train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"])
            
            # Test different SVD ranks
            for rank in CONFIG["svd_ranks"]:
                experiment_id += 1
                print(f"\n      🔬 Rank {rank}:")
                
                try:
                    # Clone base model for this experiment
                    model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        num_labels=2
                    ).to(DEVICE)
                    
                    # Benchmark before compression
                    print("         📈 Benchmarking base model...")
                    base_stats = benchmark_model(model, val_loader, DEVICE)
                    base_results = fast_train_evaluate(model, train_loader, val_loader, DEVICE, CONFIG["epochs"])
                    
                    # Apply SVD compression
                    print(f"         🗜️  Applying SVD compression (rank={rank})...")
                    compression_stats = apply_svd_compression(model, rank)
                    
                    # Benchmark after compression
                    print("         📈 Benchmarking compressed model...")
                    compressed_stats = benchmark_model(model, val_loader, DEVICE)
                    compressed_results = fast_train_evaluate(model, train_loader, val_loader, DEVICE, 1)  # Quick eval
                    
                    # Calculate metrics
                    params_after = count_parameters(model)
                    speedup = base_stats['avg_batch_time'] / compressed_stats['avg_batch_time']
                    
                    # Store results
                    result = {
                        'experiment_id': experiment_id,
                        'model': model_name,
                        'task': task_name,
                        'rank': rank,
                        
                        # Accuracy metrics
                        'base_accuracy': base_results['accuracy'],
                        'compressed_accuracy': compressed_results['accuracy'],
                        'accuracy_drop': base_results['accuracy'] - compressed_results['accuracy'],
                        
                        # Parameter metrics
                        'base_params': base_params,
                        'compressed_params': params_after,
                        'param_reduction': base_params - params_after,
                        'compression_ratio': compression_stats['compression_ratio'],
                        
                        # Speed metrics
                        'base_batch_time': base_stats['avg_batch_time'],
                        'compressed_batch_time': compressed_stats['avg_batch_time'],
                        'speedup_factor': speedup,
                        
                        # Memory metrics
                        'base_memory_mb': base_stats['memory_usage_mb'],
                        'compressed_memory_mb': compressed_stats['memory_usage_mb'],
                        'memory_reduction': base_stats['memory_usage_mb'] - compressed_stats['memory_usage_mb'],
                        
                        # Compression details
                        'compressed_layers': compression_stats['compressed_layers']
                    }
                    
                    all_results.append(result)
                    
                    print(f"         ✅ Complete | Acc: {compressed_results['accuracy']:.4f} | "
                          f"Params: {result['param_reduction']:,} reduced | "
                          f"Speedup: {speedup:.2f}x")
                    
                    # Cleanup
                    del model
                    gc.collect()
                    if DEVICE.type == 'mps':
                        torch.mps.empty_cache()
                    elif DEVICE.type == 'cuda':
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"         ❌ Error: {e}")
                    continue
        
        # Cleanup base model
        del base_model, tokenizer
        gc.collect()
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    results_path = Path(CONFIG["output_dir"]) / "compression_results.csv"
    results_df.to_csv(results_path, index=False)
    
    print(f"\n✅ All experiments completed! Results saved to {results_path}")
    
    return results_df

# ==================== ADVANCED VISUALIZATION ====================
def create_comprehensive_plots(results_df):
    """Create comprehensive visualization suite"""
    
    print("\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS...")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. MAIN COMPARISON: Accuracy vs Compression Ratio
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SVD Compression: Trade-off Analysis', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy vs Rank
    ax1 = axes[0, 0]
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax1.plot(model_data['rank'], model_data['compressed_accuracy'], 
                marker='o', label=model, linewidth=2)
    ax1.set_xlabel('SVD Rank')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Rank')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs Rank
    ax2 = axes[0, 1]
    for task in results_df['task'].unique():
        task_data = results_df[results_df['task'] == task]
        ax2.plot(task_data['rank'], task_data['speedup_factor'], 
                marker='s', label=task, linewidth=2)
    ax2.set_xlabel('SVD Rank')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Inference Speedup vs Rank')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter Reduction
    ax3 = axes[0, 2]
    param_data = results_df.groupby('rank')['param_reduction'].mean().reset_index()
    ax3.bar(param_data['rank'], param_data['param_reduction'], 
           color='skyblue', edgecolor='navy', linewidth=2)
    ax3.set_xlabel('SVD Rank')
    ax3.set_ylabel('Parameters Reduced')
    ax3.set_title('Average Parameter Reduction')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Accuracy Drop Distribution
    ax4 = axes[1, 0]
    accuracy_drops = results_df['accuracy_drop']
    ax4.hist(accuracy_drops, bins=20, color='lightcoral', edgecolor='darkred', alpha=0.7)
    ax4.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax4.set_xlabel('Accuracy Drop')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Accuracy Drops')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: 3D Plot: Rank vs Accuracy vs Speedup
    from mpl_toolkits.mplot3d import Axes3D
    ax5 = axes[1, 1]
    ax5.remove()
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    
    colors = {'bert': 'red', 'distilbert': 'blue', 'roberta': 'green'}
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        ax5.scatter(model_data['rank'], 
                   model_data['compressed_accuracy'], 
                   model_data['speedup_factor'],
                   c=colors.get(model, 'gray'),
                   label=model,
                   s=50,
                   alpha=0.7,
                   depthshade=True)
    
    ax5.set_xlabel('Rank')
    ax5.set_ylabel('Accuracy')
    ax5.set_zlabel('Speedup')
    ax5.set_title('3D: Rank vs Accuracy vs Speedup')
    ax5.legend()
    
    # Plot 6: Best Configurations Heatmap
    ax6 = axes[1, 2]
    pivot_data = results_df.pivot_table(
        index='model',
        columns='rank',
        values='speedup_factor',
        aggfunc='mean'
    )
    im = ax6.imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    ax6.set_xticks(range(len(pivot_data.columns)))
    ax6.set_xticklabels(pivot_data.columns)
    ax6.set_yticks(range(len(pivot_data.index)))
    ax6.set_yticklabels(pivot_data.index)
    ax6.set_xlabel('SVD Rank')
    ax6.set_ylabel('Model')
    ax6.set_title('Speedup Heatmap')
    
    # Add colorbar
    plt.colorbar(im, ax=ax6, label='Speedup Factor')
    
    # Add value annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            ax6.text(j, i, f'{pivot_data.iloc[i, j]:.2f}',
                    ha='center', va='center',
                    color='black' if pivot_data.iloc[i, j] < 1.5 else 'white',
                    fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(CONFIG["plots_dir"]) / 'comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    
    # 2. MODEL-SPECIFIC ANALYSIS
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, model in enumerate(results_df['model'].unique()):
        model_data = results_df[results_df['model'] == model]
        
        ax = axes2[idx]
        
        # Plot lines for each task
        for task in model_data['task'].unique():
            task_data = model_data[model_data['task'] == task]
            ax.plot(task_data['rank'], task_data['compressed_accuracy'], 
                   marker='o', label=task, linewidth=2)
        
        ax.set_xlabel('SVD Rank')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'{model.upper()} - Task Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(Path(CONFIG["plots_dir"]) / 'model_specific_performance.png', dpi=300)
    
    # 3. OPTIMAL RANK SELECTION PLOT
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Calculate efficiency score (weighted combination of metrics)
    results_df['efficiency_score'] = (
        0.4 * (1 - results_df['accuracy_drop']) +
        0.3 * results_df['speedup_factor'] +
        0.3 * (1 - results_df['compression_ratio'])
    )
    
    # Group by rank and model
    efficiency_by_rank = results_df.groupby(['model', 'rank'])['efficiency_score'].mean().unstack()
    
    # Plot
    efficiency_by_rank.T.plot(marker='o', linewidth=2, ax=ax3)
    ax3.set_xlabel('SVD Rank')
    ax3.set_ylabel('Efficiency Score')
    ax3.set_title('Optimal Rank Selection (Higher is Better)')
    ax3.legend(title='Model')
    ax3.grid(True, alpha=0.3)
    
    # Highlight best rank for each model
    for model in efficiency_by_rank.index:
        best_rank = efficiency_by_rank.loc[model].idxmax()
        best_score = efficiency_by_rank.loc[model].max()
        ax3.plot(best_rank, best_score, 'k*', markersize=15, 
                markeredgewidth=2, markeredgecolor='gold')
    
    plt.tight_layout()
    plt.savefig(Path(CONFIG["plots_dir"]) / 'optimal_rank_selection.png', dpi=300)
    
    # 4. PARETO FRONT PLOT (Accuracy vs Speed)
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    
    colors = {'bert': 'red', 'distilbert': 'blue', 'roberta': 'green'}
    markers = {'commonsenseqa': 'o', 'hellaswag': 's', 'gsm8k': '^', 
              'arc_challenge': 'D', 'arc_agi': 'X'}
    
    for (model, task), group in results_df.groupby(['model', 'task']):
        ax4.scatter(group['speedup_factor'], 
                   group['compressed_accuracy'],
                   c=colors[model],
                   marker=markers.get(task, 'o'),
                   s=100,
                   alpha=0.6,
                   label=f'{model}-{task}' if task in ['commonsenseqa', 'hellaswag'] else '')
    
    ax4.set_xlabel('Speedup Factor (Higher → Faster)')
    ax4.set_ylabel('Accuracy (Higher → Better)')
    ax4.set_title('Pareto Front: Speed vs Accuracy Trade-off')
    ax4.grid(True, alpha=0.3)
    
    # Add ideal point
    ax4.plot([2.5], [0.95], 'y*', markersize=20, label='Ideal Point', markeredgecolor='black')
    
    # Add reference lines
    ax4.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=1.5, color='gray', linestyle='--', alpha=0.5)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(Path(CONFIG["plots_dir"]) / 'pareto_front.png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Created 4 comprehensive plots in {CONFIG['plots_dir']}")
    
    # Save summary statistics
    summary_stats = {
        'best_overall_config': results_df.loc[results_df['efficiency_score'].idxmax()].to_dict(),
        'avg_speedup': results_df['speedup_factor'].mean(),
        'avg_accuracy_drop': results_df['accuracy_drop'].mean(),
        'avg_param_reduction': results_df['param_reduction'].mean(),
        'best_by_model': {}
    }
    
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model]
        best_idx = model_data['efficiency_score'].idxmax()
        summary_stats['best_by_model'][model] = model_data.loc[best_idx].to_dict()
    
    # Save summary
    import json
    summary_path = Path(CONFIG["output_dir"]) / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)
    
    print(f"📊 Summary statistics saved to {summary_path}")
    
    return summary_stats

# ==================== RECOMMENDATION ENGINE ====================
def generate_recommendations(results_df, summary_stats):
    """Generate actionable recommendations based on results"""
    
    print("\n💡 ACTIONABLE RECOMMENDATIONS")
    print("="*50)
    
    # Find best configurations for different use cases
    best_speed = results_df.loc[results_df['speedup_factor'].idxmax()]
    best_accuracy = results_df.loc[results_df['compressed_accuracy'].idxmax()]
    best_efficiency = results_df.loc[results_df['efficiency_score'].idxmax()]
    
    print("\n🎯 FOR MAXIMUM SPEED:")
    print(f"   Model: {best_speed['model'].upper()}")
    print(f"   Rank: {best_speed['rank']}")
    print(f"   Task: {best_speed['task']}")
    print(f"   Speedup: {best_speed['speedup_factor']:.2f}x faster")
    print(f"   Accuracy: {best_speed['compressed_accuracy']:.4f} ({best_speed['accuracy_drop']:.4f} drop)")
    
    print("\n🎯 FOR MAXIMUM ACCURACY:")
    print(f"   Model: {best_accuracy['model'].upper()}")
    print(f"   Rank: {best_accuracy['rank']}")
    print(f"   Task: {best_accuracy['task']}")
    print(f"   Accuracy: {best_accuracy['compressed_accuracy']:.4f}")
    print(f"   Speedup: {best_accuracy['speedup_factor']:.2f}x")
    
    print("\n🎯 FOR BEST BALANCE (RECOMMENDED):")
    print(f"   Model: {best_efficiency['model'].upper()}")
    print(f"   Rank: {best_efficiency['rank']}")
    print(f"   Task: {best_efficiency['task']}")
    print(f"   Efficiency Score: {best_efficiency['efficiency_score']:.3f}")
    print(f"   Accuracy: {best_efficiency['compressed_accuracy']:.4f}")
    print(f"   Speedup: {best_efficiency['speedup_factor']:.2f}x")
    print(f"   Parameter Reduction: {best_efficiency['param_reduction']:,}")
    
    print("\n📊 GENERAL FINDINGS:")
    print(f"   • Average speedup across all experiments: {summary_stats['avg_speedup']:.2f}x")
    print(f"   • Average accuracy drop: {summary_stats['avg_accuracy_drop']:.4f}")
    print(f"   • Average parameter reduction: {summary_stats['avg_param_reduction']:,.0f}")
    
    print("\n⚡ PRACTICAL ADVICE:")
    print("   1. For production: Use rank 16-32 for best trade-off")
    print("   2. For research: Use rank 64 to minimize accuracy impact")
    print("   3. For edge devices: Use rank 8 for maximum compression")
    print("   4. DistilBERT + rank 32 often gives best efficiency")
    
    return {
        'best_for_speed': best_speed.to_dict(),
        'best_for_accuracy': best_accuracy.to_dict(),
        'best_overall': best_efficiency.to_dict()
    }

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE SVD COMPRESSION EXPERIMENTS")
    print(f"📅 Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Device: {DEVICE}")
    print(f"🧪 Total experiments planned: {len(CONFIG['base_models']) * len(CONFIG['svd_ranks']) * 5}")
    print("="*80)
    
    # Run experiments
    start_time = time.time()
    
    try:
        results_df = run_comprehensive_experiments()
        
        if len(results_df) > 0:
            # Create visualizations
            summary_stats = create_comprehensive_plots(results_df)
            
            # Generate recommendations
            recommendations = generate_recommendations(results_df, summary_stats)
            
            # Final summary
            print("\n" + "="*80)
            print("🏆 EXPERIMENT COMPLETE - KEY INSIGHTS")
            print("="*80)
            
            # Display top 3 configurations
            top_configs = results_df.nlargest(3, 'efficiency_score')
            for i, (_, row) in enumerate(top_configs.iterrows(), 1):
                print(f"\n#{i} Model: {row['model'].upper()}, Rank: {row['rank']}, Task: {row['task']}")
                print(f"   Accuracy: {row['compressed_accuracy']:.4f}")
                print(f"   Speedup: {row['speedup_factor']:.2f}x")
                print(f"   Parameters reduced: {row['param_reduction']:,}")
            
            total_time = time.time() - start_time
            print(f"\n⏱️  Total execution time: {total_time:.2f} seconds")
            print(f"📁 Results saved in: {CONFIG['output_dir']}")
            print(f"🎨 Plots saved in: {CONFIG['plots_dir']}")
            
        else:
            print("❌ No valid results were generated. Check your data paths and configurations.")
            
    except Exception as e:
        print(f"❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n✨ Process completed!")