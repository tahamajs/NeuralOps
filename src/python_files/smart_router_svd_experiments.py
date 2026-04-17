# ==================== smart_router_svd_experiments.py ====================
"""
🌟 SMART ROUTER ENSEMBLE - COMPREHENSIVE EXPERIMENTS
Tests:
1. Different SVD ranks (8, 16, 32, 64)
2. Different training sample sizes (100, 250, 500, 750)
3. With/Without SVD comparison
4. Router vs Single Task
Author: RAINO
"""
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from config import config
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import sys
import traceback
import gc
from collections import defaultdict

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

# --- Configuration ---
NUM_LABELS = 2
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 2e-4
EPOCHS = 3

# --- Device Setup ---
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        try:
            _ = torch.zeros(1).to("mps")
            print("[CONFIG] MPS available. Using Apple Silicon GPU.")
            return torch.device("mps")
        except RuntimeError:
            return torch.device("cpu")
    return torch.device("cpu")

DEVICE = get_device()
print(f"🚀 System Ready | Device: {DEVICE}")

# --- SVD Compressor (Attention Only) ---
class SVDLinearLayer(torch.nn.Module):
    def __init__(self, linear_layer, rank):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        
        original_weight = linear_layer.weight.data
        U, S, Vh = torch.linalg.svd(original_weight, full_matrices=False)
        
        self.U = torch.nn.Parameter(U[:, :rank])
        self.S = torch.nn.Parameter(S[:rank])
        self.Vh = torch.nn.Parameter(Vh[:rank, :])
        
        if linear_layer.bias is not None:
            self.bias = torch.nn.Parameter(linear_layer.bias.data.clone())
        else:
            self.bias = None
            
    def forward(self, x):
        reconstructed = self.U @ (self.S.unsqueeze(1) * self.Vh)
        return torch.nn.functional.linear(x, reconstructed, self.bias)

def compress_attention_layers(model, rank):
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            name_lower = name.lower()
            is_attention = (
                'attention' in name_lower and (
                    'query' in name_lower or 
                    'key' in name_lower or 
                    'value' in name_lower or 
                    'dense' in name_lower or
                    'output' in name_lower
                )
            )
            if is_attention:
                try:
                    new_layer = SVDLinearLayer(module, rank)
                    parts = name.split('.')
                    if len(parts) > 1:
                        parent_name = '.'.join(parts[:-1])
                        child_name = parts[-1]
                        parent = model.get_submodule(parent_name)
                        setattr(parent, child_name, new_layer)
                        count += 1
                except:
                    pass
    return count

# --- Dataset ---
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        encoding = self.tokenizer(
            text, truncation=True, padding='max_length', 
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Router ---
ROUTER_KEYWORDS = {
    'comparative': ['better', 'worse', 'than', 'more', 'less', 'higher', 'lower', 'superior', 'inferior', 'best', 'worst', 'prefer'],
    'negation': ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nor', 'without', "n't", 'dont', "don't", 'wont', "won't", 'cant', "can't"],
    'irony': []
}

def detect_task_type(text):
    text_lower = text.lower()
    comp_count = sum(1 for kw in ROUTER_KEYWORDS['comparative'] if kw in text_lower)
    neg_count = sum(1 for kw in ROUTER_KEYWORDS['negation'] if kw in text_lower)
    
    if comp_count > neg_count and comp_count > 0:
        return 'comparative'
    elif neg_count > 0:
        return 'negation'
    return 'irony'

# --- Training Function ---
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    best_f1 = 0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        # Validation
        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                true_labels.extend(batch['labels'].numpy())
        
        val_f1 = f1_score(true_labels, preds, average='weighted')
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
    
    return best_f1

def evaluate_model(model, val_loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
            true_labels.extend(batch['labels'].numpy())
    
    return accuracy_score(true_labels, preds), f1_score(true_labels, preds, average='weighted')

# --- Main Experiment ---
def run_experiments():
    print("="*80)
    print("🌟 SMART ROUTER - COMPREHENSIVE EXPERIMENTS")
    print(f"Device: {DEVICE}")
    print("="*80)
    
    base_model = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Experiment configurations
    SVD_RANKS = [8, 16, 32, 64]
    TRAIN_SAMPLES_LIST = [100, 250, 500, 750]
    
    data_files = {
        "01_irony": "./data/01_irony.csv",
        "02_negation": "./data/02_negation.csv",
        "03_comparative": "./data/03_comparative.csv"
    }
    
    # Store results
    results = {
        'rank_experiment': defaultdict(list),
        'samples_experiment': defaultdict(list),
        'svd_comparison': {},
        'router_breakdown': {}
    }
    
    # ==================== EXPERIMENT 1: SVD Rank Comparison ====================
    print("\n" + "="*60)
    print("📊 EXPERIMENT 1: SVD Rank Comparison")
    print("="*60)
    
    for rank in SVD_RANKS:
        print(f"\n🔬 Testing SVD Rank = {rank}")
        
        for task_name in ["01_irony", "02_negation", "03_comparative"]:
            data_path = data_files[task_name]
            if not Path(data_path).exists():
                continue
            
            df = pd.read_csv(data_path)
            t_col = [c for c in df.columns if 'text' in c.lower()][0]
            l_col = [c for c in df.columns if 'label' in c.lower()][0]
            df = df.rename(columns={t_col: 'text', l_col: 'label'})
            
            # Use 500 samples
            df_train = df.head(500)
            train_df = df_train.iloc[:400].reset_index(drop=True)
            val_df = df_train.iloc[400:].reset_index(drop=True)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=getattr(config, 'hf_cache_dir', None))
                model = AutoModelForSequenceClassification.from_pretrained(
                    base_model, num_labels=NUM_LABELS, cache_dir=getattr(config, 'hf_cache_dir', None)
                ).to(DEVICE)
                
                train_dataset = SentimentDataset(train_df['text'], train_df['label'], tokenizer)
                val_dataset = SentimentDataset(val_df['text'], val_df['label'], tokenizer)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
                
                # Train
                train_model(model, train_loader, val_loader)
                
                # Evaluate before SVD
                acc_before, f1_before = evaluate_model(model, val_loader)
                
                # Apply SVD
                compress_attention_layers(model, rank)
                
                # Evaluate after SVD
                acc_after, f1_after = evaluate_model(model, val_loader)
                
                results['rank_experiment'][f'rank_{rank}'].append({
                    'task': task_name,
                    'acc_before': acc_before,
                    'acc_after': acc_after,
                    'f1_before': f1_before,
                    'f1_after': f1_after,
                    'svd_impact': acc_after - acc_before
                })
                
                print(f"   {task_name}: Pre-SVD={acc_before:.4f} → Post-SVD={acc_after:.4f} (Δ={acc_after-acc_before:+.4f})")
                
                del model
                gc.collect()
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # ==================== EXPERIMENT 2: Training Samples ====================
    print("\n" + "="*60)
    print("📊 EXPERIMENT 2: Training Samples Comparison")
    print("="*60)
    
    for n_samples in TRAIN_SAMPLES_LIST:
        print(f"\n🔬 Testing with {n_samples} training samples")
        
        for task_name in ["01_irony", "02_negation", "03_comparative"]:
            data_path = data_files[task_name]
            if not Path(data_path).exists():
                continue
            
            df = pd.read_csv(data_path)
            t_col = [c for c in df.columns if 'text' in c.lower()][0]
            l_col = [c for c in df.columns if 'label' in c.lower()][0]
            df = df.rename(columns={t_col: 'text', l_col: 'label'})
            
            df_train = df.head(n_samples)
            train_size = int(0.8 * len(df_train))
            train_df = df_train.iloc[:train_size].reset_index(drop=True)
            val_df = df_train.iloc[train_size:].reset_index(drop=True)
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=getattr(config, 'hf_cache_dir', None))
                model = AutoModelForSequenceClassification.from_pretrained(
                    base_model, num_labels=NUM_LABELS, cache_dir=getattr(config, 'hf_cache_dir', None)
                ).to(DEVICE)
                
                train_dataset = SentimentDataset(train_df['text'], train_df['label'], tokenizer)
                val_dataset = SentimentDataset(val_df['text'], val_df['label'], tokenizer)
                train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
                
                # Train with SVD from start
                compress_attention_layers(model, rank=32)
                train_model(model, train_loader, val_loader)
                
                acc, f1 = evaluate_model(model, val_loader)
                
                results['samples_experiment'][f'samples_{n_samples}'].append({
                    'task': task_name,
                    'accuracy': acc,
                    'f1': f1
                })
                
                print(f"   {task_name}: Acc={acc:.4f}, F1={f1:.4f}")
                
                del model
                gc.collect()
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
    
    # ==================== EXPERIMENT 3: SVD vs No-SVD ====================
    print("\n" + "="*60)
    print("📊 EXPERIMENT 3: SVD vs No-SVD Comparison")
    print("="*60)
    
    for task_name in ["01_irony", "02_negation", "03_comparative"]:
        data_path = data_files[task_name]
        if not Path(data_path).exists():
            continue
        
        df = pd.read_csv(data_path)
        t_col = [c for c in df.columns if 'text' in c.lower()][0]
        l_col = [c for c in df.columns if 'label' in c.lower()][0]
        df = df.rename(columns={t_col: 'text', l_col: 'label'})
        
        df_train = df.head(500)
        train_df = df_train.iloc[:400].reset_index(drop=True)
        val_df = df_train.iloc[400:].reset_index(drop=True)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=getattr(config, 'hf_cache_dir', None))
            
            # Without SVD
            model_no_svd = AutoModelForSequenceClassification.from_pretrained(
                base_model, num_labels=NUM_LABELS, cache_dir=getattr(config, 'hf_cache_dir', None)
            ).to(DEVICE)
            
            train_dataset = SentimentDataset(train_df['text'], train_df['label'], tokenizer)
            val_dataset = SentimentDataset(val_df['text'], val_df['label'], tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            train_model(model_no_svd, train_loader, val_loader)
            acc_no_svd, f1_no_svd = evaluate_model(model_no_svd, val_loader)
            
            del model_no_svd
            gc.collect()
            
            # With SVD
            model_svd = AutoModelForSequenceClassification.from_pretrained(
                base_model, num_labels=NUM_LABELS, cache_dir=getattr(config, 'hf_cache_dir', None)
            ).to(DEVICE)
            
            train_model(model_svd, train_loader, val_loader)
            compress_attention_layers(model_svd, rank=32)
            acc_svd, f1_svd = evaluate_model(model_svd, val_loader)
            
            results['svd_comparison'][task_name] = {
                'no_svd': {'acc': acc_no_svd, 'f1': f1_no_svd},
                'svd': {'acc': acc_svd, 'f1': f1_svd}
            }
            
            print(f"   {task_name}: No-SVD={acc_no_svd:.4f}, With-SVD={acc_svd:.4f} (Δ={acc_svd-acc_no_svd:+.4f})")
            
            del model_svd
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # ==================== EXPERIMENT 4: Router Full Evaluation ====================
    print("\n" + "="*60)
    print("📊 EXPERIMENT 4: Router Full Evaluation")
    print("="*60)
    
    # Train final models for router
    router_models = {}
    
    for task_name in ["01_irony", "02_negation", "03_comparative"]:
        data_path = data_files[task_name]
        if not Path(data_path).exists():
            continue
        
        df = pd.read_csv(data_path)
        t_col = [c for c in df.columns if 'text' in c.lower()][0]
        l_col = [c for c in df.columns if 'label' in c.lower()][0]
        df = df.rename(columns={t_col: 'text', l_col: 'label'})
        
        df_train = df.head(500)
        train_df = df_train.iloc[:400].reset_index(drop=True)
        val_df = df_train.iloc[400:].reset_index(drop=True)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model, cache_dir=getattr(config, 'hf_cache_dir', None))
            model = AutoModelForSequenceClassification.from_pretrained(
                base_model, num_labels=NUM_LABELS, cache_dir=getattr(config, 'hf_cache_dir', None)
            ).to(DEVICE)
            
            train_dataset = SentimentDataset(train_df['text'], train_df['label'], tokenizer)
            val_dataset = SentimentDataset(val_df['text'], val_df['label'], tokenizer)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
            
            train_model(model, train_loader, val_loader)
            compress_attention_layers(model, rank=32)
            
            router_models[task_name] = {'model': model, 'tokenizer': tokenizer}
            
            del model
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Evaluate router on full data
    all_texts, all_true_labels, all_source_tasks = [], [], []
    
    for task_name in router_models.keys():
        full_df = pd.read_csv(data_files[task_name])
        t_col = [c for c in full_df.columns if 'text' in c.lower()][0]
        l_col = [c for c in full_df.columns if 'label' in c.lower()][0]
        full_df = full_df.rename(columns={t_col: 'text', l_col: 'label'})
        
        all_texts.extend(full_df['text'].tolist())
        all_true_labels.extend(full_df['label'].tolist())
        all_source_tasks.extend([task_name] * len(full_df))
    
    print(f"   Total samples: {len(all_texts)}")
    
    # Run router
    router_preds = []
    for i, text in enumerate(all_texts):
        detected = detect_task_type(text)
        if detected not in router_models:
            detected = '01_irony'
        
        model = router_models[detected]['model']
        tokenizer = router_models[detected]['tokenizer']
        
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=MAX_LENGTH, return_tensors='pt')
        
        with torch.no_grad():
            outputs = model(input_ids=encoding['input_ids'].to(DEVICE), attention_mask=encoding['attention_mask'].to(DEVICE))
            pred = torch.argmax(outputs.logits, dim=-1).item()
        
        router_preds.append(pred)
    
    # Calculate metrics
    final_acc = accuracy_score(all_true_labels, router_preds)
    final_f1 = f1_score(all_true_labels, router_preds, average='weighted')
    
    results['router_breakdown'] = {
        'overall': {'accuracy': final_acc, 'f1': final_f1}
    }
    
    # Per-task breakdown
    for task_name in router_models.keys():
        task_indices = [i for i, src in enumerate(all_source_tasks) if src == task_name]
        t_preds = [router_preds[i] for i in task_indices]
        t_labels = [all_true_labels[i] for i in task_indices]
        t_acc = accuracy_score(t_labels, t_preds)
        t_f1 = f1_score(t_labels, t_preds, average='weighted')
        results['router_breakdown'][task_name] = {'accuracy': t_acc, 'f1': t_f1, 'count': len(task_indices)}
    
    # ==================== GENERATE PLOTS ====================
    print("\n" + "="*60)
    print("📈 GENERATING PLOTS...")
    print("="*60)
    
    generate_plots(results)
    
    print("\n✅ Experiments completed!")
    
    return results

def generate_plots(results):
    """Generate all plots"""
    
    # Plot 1: SVD Rank vs Accuracy
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1.1 SVD Rank Comparison
    ax1 = axes[0, 0]
    ranks = [8, 16, 32, 64]
    tasks = ["01_irony", "02_negation", "03_comparative"]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    x = np.arange(len(ranks))
    width = 0.25
    
    for i, task in enumerate(tasks):
        acc_before = [results['rank_experiment'][f'rank_{r}'][i]['acc_before'] for r in ranks]
        ax1.bar(x + i*width, acc_before, width, label=f'{task} (Pre-SVD)', color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('SVD Rank')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Pre-SVD Accuracy by Rank')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(ranks)
    ax1.legend()
    ax1.set_ylim(0, 1)
    
    # 1.2 Post-SVD Accuracy
    ax2 = axes[0, 1]
    for i, task in enumerate(tasks):
        acc_after = [results['rank_experiment'][f'rank_{r}'][i]['acc_after'] for r in ranks]
        ax2.bar(x + i*width, acc_after, width, label=f'{task} (Post-SVD)', color=colors[i], alpha=0.7)
    
    ax2.set_xlabel('SVD Rank')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Post-SVD Accuracy by Rank')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(ranks)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # 1.3 SVD Impact (Accuracy Drop)
    ax3 = axes[1, 0]
    for i, task in enumerate(tasks):
        svd_impact = [results['rank_experiment'][f'rank_{r}'][i]['svd_impact'] for r in ranks]
        ax3.plot(ranks, svd_impact, 'o-', label=task, color=colors[i], linewidth=2, markersize=8)
    
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('SVD Rank')
    ax3.set_ylabel('Accuracy Change')
    ax3.set_title('SVD Impact (Post-SVD - Pre-SVD)')
    ax3.legend()
    
    # 1.4 Training Samples vs Accuracy
    ax4 = axes[1, 1]
    sample_sizes = [100, 250, 500, 750]
    for i, task in enumerate(tasks):
        accs = [results['samples_experiment'][f'samples_{s}'][i]['accuracy'] for s in sample_sizes]
        ax4.plot(sample_sizes, accs, 'o-', label=task, color=colors[i], linewidth=2, markersize=8)
    
    ax4.set_xlabel('Training Samples')
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy vs Training Samples (with SVD)')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('./plots/experiment_rank_samples.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: experiment_rank_samples.png")
    plt.close()
    
    # Plot 2: SVD vs No-SVD Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tasks = list(results['svd_comparison'].keys())
    no_svd_acc = [results['svd_comparison'][t]['no_svd']['acc'] for t in tasks]
    svd_acc = [results['svd_comparison'][t]['svd']['acc'] for t in tasks]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, no_svd_acc, width, label='Without SVD', color='#2ECC71', alpha=0.8)
    bars2 = ax.bar(x + width/2, svd_acc, width, label='With SVD (rank=32)', color='#E74C3C', alpha=0.8)
    
    ax.set_xlabel('Task')
    ax.set_ylabel('Accuracy')
    ax.set_title('SVD vs No-SVD Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('./plots/experiment_svd_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: experiment_svd_comparison.png")
    plt.close()
    
    # Plot 3: Router Performance Breakdown
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 3.1 Bar chart
    ax1 = axes[0]
    task_names = list(results['router_breakdown'].keys())
    task_names = [t for t in task_names if t != 'overall']
    accs = [results['router_breakdown'][t]['accuracy'] for t in task_names]
    counts = [results['router_breakdown'][t]['count'] for t in task_names]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(task_names, accs, color=colors, alpha=0.8)
    
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Router Accuracy by Task')
    ax1.set_ylim(0, 1)
    
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.annotate(f'{height:.3f}\n(n={count})', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # 3.2 Overall metrics
    ax2 = axes[1]
    overall_acc = results['router_breakdown']['overall']['accuracy']
    overall_f1 = results['router_breakdown']['overall']['f1']
    
    metrics = ['Accuracy', 'F1 Score']
    values = [overall_acc, overall_f1]
    colors = ['#9B59B6', '#F39C12']
    
    bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
    ax2.set_ylabel('Score')
    ax2.set_title('Router Overall Performance')
    ax2.set_ylim(0, 1)
    
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('./plots/experiment_router_breakdown.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: experiment_router_breakdown.png")
    plt.close()
    
    # Plot 4: Heatmap - SVD Impact by Rank and Task
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ranks = [8, 16, 32, 64]
    tasks = ["01_irony", "02_negation", "03_comparative"]
    
    impact_matrix = []
    for r in ranks:
        row = [results['rank_experiment'][f'rank_{r}'][i]['svd_impact'] for i in range(len(tasks))]
        impact_matrix.append(row)
    
    impact_matrix = np.array(impact_matrix)
    
    sns.heatmap(impact_matrix, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                xticklabels=tasks, yticklabels=ranks, ax=ax)
    ax.set_xlabel('Task')
    ax.set_ylabel('SVD Rank')
    ax.set_title('SVD Impact Heatmap (Accuracy Change)')
    
    plt.tight_layout()
    plt.savefig('./plots/experiment_heatmap.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: experiment_heatmap.png")
    plt.close()
    
    # Plot 5: Summary Dashboard
    fig = plt.figure(figsize=(16, 10))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 5.1 Best rank per task
    ax1 = fig.add_subplot(gs[0, 0])
    best_ranks = []
    for i, task in enumerate(tasks):
        rank_accs = [(r, results['rank_experiment'][f'rank_{r}'][i]['acc_after']) for r in ranks]
        best_rank = max(rank_accs, key=lambda x: x[1])[0]
        best_ranks.append(best_rank)
    ax1.bar(tasks, best_ranks, color=colors, alpha=0.8)
    ax1.set_ylabel('Best Rank')
    ax1.set_title('Best SVD Rank per Task')
    ax1.set_ylim(0, 70)
    
    # 5.2 Average accuracy by rank
    ax2 = fig.add_subplot(gs[0, 1])
    avg_accs = []
    for r in ranks:
        accs = [results['rank_experiment'][f'rank_{r}'][i]['acc_after'] for i in range(len(tasks))]
        avg_accs.append(np.mean(accs))
    ax2.plot(ranks, avg_accs, 'o-', linewidth=2, markersize=10, color='#9B59B6')
    ax2.fill_between(ranks, avg_accs, alpha=0.3, color='#9B59B6')
    ax2.set_xlabel('SVD Rank')
    ax2.set_ylabel('Average Accuracy')
    ax2.set_title('Average Accuracy vs Rank')
    
    # 5.3 Training curve
    ax3 = fig.add_subplot(gs[0, 2])
    sample_sizes = [100, 250, 500, 750]
    avg_accs_samples = []
    for s in sample_sizes:
        accs = [results['samples_experiment'][f'samples_{s}'][i]['accuracy'] for i in range(len(tasks))]
        avg_accs_samples.append(np.mean(accs))
    ax3.plot(sample_sizes, avg_accs_samples, 's-', linewidth=2, markersize=10, color='#E74C3C')
    ax3.fill_between(sample_sizes, avg_accs_samples, alpha=0.3, color='#E74C3C')
    ax3.set_xlabel('Training Samples')
    ax3.set_ylabel('Average Accuracy')
    ax3.set_title('Learning Curve')
    
    # 5.4 SVD savings
    ax4 = fig.add_subplot(gs[1, 0])
    svd_savings = []
    for task in tasks:
        no_svd = results['svd_comparison'][task]['no_svd']['acc']
        with_svd = results['svd_comparison'][task]['svd']['acc']
        savings = (no_svd - with_svd) / no_svd * 100
        svd_savings.append(savings)
    ax4.bar(tasks, svd_savings, color=['#E74C3C', '#F39C12', '#2ECC71'], alpha=0.8)
    ax4.set_ylabel('Accuracy Drop (%)')
    ax4.set_title('SVD Cost (Accuracy Drop)')
    
    # 5.5 Router vs Best Single
    ax5 = fig.add_subplot(gs[1, 1])
    router_acc = results['router_breakdown']['overall']['accuracy']
    best_single = max([results['svd_comparison'][t]['svd']['acc'] for t in tasks])
    ax5.bar(['Router', 'Best Single'], [router_acc, best_single], color=['#3498DB', '#1ABC9C'], alpha=0.8)
    ax5.set_ylabel('Accuracy')
    ax5.set_title('Router vs Best Single Task')
    ax5.set_ylim(0, 1)
    
    # 5.6 Final summary
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary_text = f"""
    📊 EXPERIMENT SUMMARY
    ─────────────────────
    Best SVD Rank: {int(np.mean(best_ranks))}
    Avg Accuracy (Rank 32): {avg_accs[2]:.4f}
    Avg Accuracy (750 samples): {avg_accs_samples[-1]:.4f}
    Router Accuracy: {router_acc:.4f}
    Router F1: {results['router_breakdown']['overall']['f1']:.4f}
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center', 
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 5.7 Task distribution
    ax7 = fig.add_subplot(gs[2, :2])
    task_counts = [results['router_breakdown'][t]['count'] for t in tasks]
    ax7.pie(task_counts, labels=tasks, autopct='%1.1f%%', colors=colors, explode=[0.02]*3)
    ax7.set_title('Test Data Distribution')
    
    # 5.8 Performance by task
    ax8 = fig.add_subplot(gs[2, 2])
    task_f1s = [results['router_breakdown'][t]['f1'] for t in tasks]
    ax8.barh(tasks, task_f1s, color=colors, alpha=0.8)
    ax8.set_xlabel('F1 Score')
    ax8.set_title('Router F1 by Task')
    ax8.set_xlim(0, 1)
    
    plt.suptitle('🌟 SMART ROUTER EXPERIMENT DASHBOARD', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig('./plots/experiment_dashboard.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: experiment_dashboard.png")
    plt.close()
    
    print("\n📁 All plots saved to ./plots/")

# --- Main ---
if __name__ == "__main__":
    # Create plots directory
    Path("./plots").mkdir(exist_ok=True)
    
    try:
        results = run_experiments()
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        traceback.print_exc()
        sys.exit(1)