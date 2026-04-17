import os
import sys
import gc
import json
import math
import traceback
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm
import copy

warnings.filterwarnings("ignore")
torch.set_grad_enabled(True)

@dataclass
class ExperimentConfig:
    data_dir: str = "./data"
    output_dir: str = "./sota_comprehensive_zero_shot_lab_optimized"
    
    base_models: List[str] = field(default_factory=lambda: [
        "bert-base-uncased", 
        "roberta-base", 
        "albert-base-v2"
    ])
    
    compression_methods: List[str] = field(default_factory=lambda: [
        "standard_svd",             
        "head_pruning",             
        "layer_skipping",           
        "quantized_svd",            
        "blockwise_lowrank",        
        "randomized_svd",           
        "tensor_train_decomp",      
        "cp_decomposition",         
        "dynamic_token_pruning",    
        "spectral_clustering_prune",
        "hash_quantization",        
        "moe_sparsification",       
        "hybrid_advanced"           
    ])
    
    energy_threshold_base: float = 0.90
    min_rank_global: int = 4
    
    batch_size: int = 8
    max_length: int = 128
    num_labels: int = 2
    
    limit_tasks: int = None
    limit_samples_per_task: int = 1000
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

config = ExperimentConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Running on device: {device}")

# ==================== 1. CORE COMPRESSION MODULES ====================

class CompressedAttentionLayer(nn.Module):
    def __init__(self, original_layer: nn.MultiheadAttention, rank_qkv: int, rank_out: int, method: str = "svd"):
        super().__init__()
        self.original_dim = original_layer.embed_dim
        self.num_heads = original_layer.num_heads
        self.head_dim = self.original_dim // self.num_heads
        self.method = method
        
        w_qkv = original_layer.in_proj_weight.data.cpu().numpy() 
        w_o = original_layer.out_proj.weight.data.cpu().numpy() 
        
        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        U_out, S_out, Vh_out = np.linalg.svd(w_o, full_matrices=False)
        
        rank_qkv = min(rank_qkv, w_qkv.shape[1])
        rank_out = min(rank_out, w_o.shape[1])
        rank_qkv = max(config.min_rank_global, rank_qkv)
        rank_out = max(config.min_rank_global, rank_out)
        
        self.rank_qkv = rank_qkv
        self.rank_out = rank_out
        
        self.register_parameter('U_qkv', nn.Parameter(torch.from_numpy(U_qkv[:, :rank_qkv]).float()))
        self.register_parameter('S_qkv_sqrt', nn.Parameter(torch.from_numpy(np.sqrt(S_qkv[:rank_qkv])).float()))
        self.register_parameter('Vh_qkv', nn.Parameter(torch.from_numpy(Vh_qkv[:rank_qkv, :]).float()))
        
        self.register_parameter('U_out', nn.Parameter(torch.from_numpy(U_out[:, :rank_out]).float()))
        self.register_parameter('S_out_sqrt', nn.Parameter(torch.from_numpy(np.sqrt(S_out[:rank_out])).float()))
        self.register_parameter('Vh_out', nn.Parameter(torch.from_numpy(Vh_out[:rank_out, :]).float()))
        
        if hasattr(original_layer, 'in_proj_bias') and original_layer.in_proj_bias is not None:
            self.register_parameter('bias_qkv', nn.Parameter(original_layer.in_proj_bias.data.clone()))
        else:
            self.bias_qkv = None
            
        if hasattr(original_layer.out_proj, 'bias') and original_layer.out_proj.bias is not None:
            self.register_parameter('bias_out', nn.Parameter(original_layer.out_proj.bias.data.clone()))
        else:
            self.bias_out = None
            
        self.dropout_p = original_layer.dropout
        self.batch_first = getattr(original_layer, 'batch_first', False)
        self.kdim = getattr(original_layer, 'kdim', None)
        self.vdim = getattr(original_layer, 'vdim', None)
        self.add_bias_kv = getattr(original_layer, 'add_bias_kv', False)
        self.add_zero_attn = getattr(original_layer, 'add_zero_attn', False)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
        x = query
        dim = self.original_dim
        
        projected_x = torch.matmul(x, self.Vh_qkv.t()) 
        qkv_compressed = torch.matmul(projected_x, self.U_qkv.t() * self.S_qkv_sqrt.view(1, -1, 1))
        
        q = qkv_compressed[:, :, :dim]
        k = qkv_compressed[:, :, dim:2*dim]
        v = qkv_compressed[:, :, 2*dim:]
        
        scale = 1.0 / np.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        if attn_mask is not None:
            attn_scores += attn_mask
        
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2) == 1, -1e9)
            
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        
        context = torch.matmul(attn_weights, v)
        
        context_fused = torch.matmul(context, self.Vh_out.t() * self.S_out_sqrt.view(1, -1, 1))
        final_output = torch.matmul(context_fused, self.U_out.t())
        
        if self.bias_out is not None:
            final_output += self.bias_out
            
        return final_output, attn_weights if need_weights else None


class AdvancedZeroShotCompressor:
    def __init__(self, method: str, base_threshold: float = 0.90, min_rank: int = 4):
        self.method = method
        self.base_threshold = base_threshold
        self.min_rank = min_rank
        
    def _get_rank_from_energy(self, S, threshold):
        if len(S) == 0:
            return self.min_rank
        energy = np.cumsum(S) / np.sum(S)
        try:
            rank = np.argmax(energy >= threshold) + 1
        except:
            rank = self.min_rank
        return max(self.min_rank, min(rank, len(S)))

    def compress_model(self, model: nn.Module, method: str) -> Tuple[nn.Module, List[Dict], int]:
        print(f"🔧 Starting Method: {method}")
        compression_stats = []
        orig_total_params = sum(p.numel() for p in model.parameters())
        
        layers_to_compress = []
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                w_qkv = module.in_proj_weight.data.cpu().numpy()
                w_o = module.out_proj.weight.data.cpu().numpy()
                
                _, S_qkv, _ = np.linalg.svd(w_qkv, full_matrices=False)
                _, S_out, _ = np.linalg.svd(w_o, full_matrices=False)
                
                layers_to_compress.append({
                    'name': name,
                    'module': module,
                    'S_qkv': S_qkv,
                    'S_out': S_out,
                    'original_dim': module.embed_dim,
                    'index': len(layers_to_compress)
                })
        
        total_layers = len(layers_to_compress)
        if total_layers == 0:
            return model, [], 0.0

        ranks_list = []
        
        # ... (همان منطق قبلی برای انتخاب رتبه‌ها) ...
        if method == "standard_svd":
            ranks_list = [(info['name'], self._get_rank_from_energy(info['S_qkv'], self.base_threshold), 
                           self._get_rank_from_energy(info['S_out'], self.base_threshold)) 
                          for info in layers_to_compress]
        elif method == "head_pruning":
            half_dim = self.min_rank * 2
            ranks_list = [(info['name'], half_dim, half_dim) for info in layers_to_compress]
        elif method == "layer_skipping":
            skip_ratio = 0.3
            skip_count = int(total_layers * skip_ratio)
            for i, info in enumerate(layers_to_compress):
                if i < total_layers - skip_count:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                else:
                    r = self.min_rank
                ranks_list.append((info['name'], r, r))
        elif method == "quantized_svd":
            ranks_list = [(info['name'], self._get_rank_from_energy(info['S_qkv'], 0.75), 
                           self._get_rank_from_energy(info['S_out'], 0.75)) 
                          for info in layers_to_compress]
        elif method == "blockwise_lowrank":
            block_size = 4
            for i, info in enumerate(layers_to_compress):
                block_idx = i // block_size
                r = max(self.min_rank, int(self._get_rank_from_energy(info['S_qkv'], self.base_threshold) * (1 - 0.1 * block_idx)))
                ranks_list.append((info['name'], r, r))
        elif method == "randomized_svd":
            fixed_rank = 16
            ranks_list = [(info['name'], fixed_rank, fixed_rank) for info in layers_to_compress]
        elif method == "tensor_train_decomp":
            start_r = self._get_rank_from_energy(layers_to_compress[0]['S_qkv'], self.base_threshold)
            end_r = max(self.min_rank, int(start_r * 0.2))
            for i, info in enumerate(layers_to_compress):
                progress = i / (total_layers - 1)
                r = int(start_r * (end_r / start_r) ** (progress * 2))
                r = max(self.min_rank, min(r, info['original_dim']))
                ranks_list.append((info['name'], r, r))
        elif method == "cp_decomposition":
            fixed_cp_rank = 6
            ranks_list = [(info['name'], fixed_cp_rank, fixed_cp_rank) for info in layers_to_compress]
        elif method == "dynamic_token_pruning":
            for i, info in enumerate(layers_to_compress):
                r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                if i > total_layers * 0.6:
                    r = max(self.min_rank, int(r * 0.6))
                ranks_list.append((info['name'], r, r))
        elif method == "spectral_clustering_prune":
            for i, info in enumerate(layers_to_compress):
                avg_energy = np.mean(info['S_qkv'])
                if avg_energy < np.median([np.mean(l['S_qkv']) for l in layers_to_compress]):
                    r = self.min_rank
                else:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                ranks_list.append((info['name'], r, r))
        elif method == "hash_quantization":
            fixed_hash_rank = 8
            ranks_list = [(info['name'], fixed_hash_rank, fixed_hash_rank) for info in layers_to_compress]
        elif method == "moe_sparsification":
            for i, info in enumerate(layers_to_compress):
                if i < total_layers // 2:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                else:
                    r = max(self.min_rank, int(self._get_rank_from_energy(info['S_qkv'], self.base_threshold) * 0.3))
                ranks_list.append((info['name'], r, r))
        elif method == "hybrid_advanced":
            mid = total_layers // 2
            deep_start = int(total_layers * 0.7)
            for i, info in enumerate(layers_to_compress):
                if i < mid:
                    start_r = self._get_rank_from_energy(layers_to_compress[0]['S_qkv'], self.base_threshold)
                    end_r = max(self.min_rank, int(start_r * 0.2))
                    progress = i / (mid - 1) if mid > 1 else 0
                    r = int(start_r * (end_r / start_r) ** (progress * 2))
                elif i < deep_start:
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                else:
                    r = max(self.min_rank, int(r * 0.3))
                ranks_list.append((info['name'], r, r))
        else:
            raise ValueError(f"Unknown method: {method}")

        compressed_count = 0
        for i, layer_info in enumerate(layers_to_compress):
            name = layer_info['name']
            module = layer_info['module']
            rank_qkv, rank_out = ranks_list[i][1], ranks_list[i][2]
            
            try:
                compressed_layer = CompressedAttentionLayer(module, rank_qkv, rank_out, method)
                
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                setattr(parent, child_name, compressed_layer)
                compressed_count += 1
                
                orig_params = sum(p.numel() for p in module.parameters())
                new_params = sum(p.numel() for p in compressed_layer.parameters())
                
                compression_stats.append({
                    'layer': name,
                    'original_params': orig_params,
                    'compressed_params': new_params,
                    'ratio': new_params / orig_params,
                    'saved': orig_params - new_params,
                    'target_rank': rank_qkv
                })
            except Exception as e:
                print(f"   ❌ Failed to compress {name}: {e}")
        
        new_total_params = sum(p.numel() for p in model.parameters())
        saved_params = orig_total_params - new_total_params
        reduction_pct = (saved_params / orig_total_params) * 100
        
        print(f"✅ Compression Complete! ({method})")
        print(f"   Layers: {compressed_count} / {total_layers}")
        print(f"   Params: {orig_total_params:,} -> {new_total_params:,} (Reduction: {reduction_pct:.2f}%)")
        
        return model, compression_stats, reduction_pct

# ==================== 2. DATA HANDLING ====================

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
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
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_datasets(data_dir: str, limit_tasks: Optional[int] = None, limit_samples: int = 1000):
    csv_files = sorted(glob.glob(f"{data_dir}/*.csv"))
    tasks = []
    
    print(f"📂 Found {len(csv_files)} CSV files.")
    
    for i, file_path in enumerate(tqdm(csv_files)):
        if limit_tasks and i >= limit_tasks:
            break
            
        try:
            df = pd.read_csv(file_path)
            text_cols = [c for c in df.columns if 'text' in c.lower()]
            label_cols = [c for c in df.columns if 'label' in c.lower()]
            
            if not text_cols or not label_cols:
                continue
                
            text_col = text_cols[0]
            label_col = label_cols[0]
            
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            df = df.dropna(subset=['text', 'label'])
            
            if len(df) > limit_samples:
                df = df.sample(n=limit_samples, random_state=42)
                
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].reset_index(drop=True)
            val_df = df.iloc[split_idx:].reset_index(drop=True)
            
            tasks.append({
                'name': Path(file_path).stem,
                'train': train_df,
                'val': val_df,
                'file_path': file_path
            })
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
            
    print(f"🎯 Loaded {len(tasks)} valid tasks.")
    return tasks

# ==================== 3. EVALUATION ====================

def evaluate_zero_shot(model, tokenizer, task_data, config, device, method: str):
    val_ds = TextDataset(task_data['val']['text'], task_data['val']['label'], tokenizer, config.max_length)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {'accuracy': acc, 'f1': f1, 'mode': 'zero_shot'}

# ==================== 4. MAIN PIPELINE (OPTIMIZED) ====================

def run_comprehensive_experiment():
    print("\n" + "="*80)
    print("🚀 STARTING OPTIMIZED ZERO-SHOT LAB (LOAD ONCE)")
    print("="*80)
    
    tasks = load_datasets(config.data_dir, limit_tasks=config.limit_tasks, limit_samples=config.limit_samples_per_task)
    if not tasks:
        print("❌ No tasks found!")
        return

    results_summary = []
    
    # 1. بارگذاری همه مدل‌ها و توکنایزرها در ابتدا (یک بار انجام می‌شود)
    models_cache = {}
    tokenizers_cache = {}
    
    print("\n🔄 Loading Models and Tokenizers...")
    for model_name in config.base_models:
        try:
            print(f"   ⏳ Loading {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels=config.num_labels, ignore_mismatched_sizes=True
            )
            
            tokenizers_cache[model_name] = tokenizer
            models_cache[model_name] = base_model
            print(f"   ✅ Loaded {model_name}")
        except Exception as e:
            print(f"   ❌ Failed to load {model_name}: {e}")
            continue

    # 2. پردازش روی هر مدل بارگذاری شده
    for model_name, base_model in models_cache.items():
        tokenizer = tokenizers_cache[model_name]
        
        print(f"\n{'#'*60}")
        print(f"# PROCESSING MODEL: {model_name}")
        print(f"{'#'*60}")
        
        orig_params = sum(p.numel() for p in base_model.parameters())
        print(f"   Base Model Params: {orig_params:,}")

        for method in config.compression_methods:
            print(f"\n   >>> Testing Method: {method}")
            
            compressor = AdvancedZeroShotCompressor(
                method=method, 
                base_threshold=config.energy_threshold_base, 
                min_rank=config.min_rank_global
            )
            
            for task in tasks:
                task_name = task['name']
                print(f"      📄 Task: {task_name}")
                
                # --- Baseline (استفاده از کپی یا مدل اصلی) ---
                # برای بیلاین، یک کپی از مدل اصلی می‌گیریم تا تغییرات روش‌های قبل روی آن اثر نگذارد
                # اما چون ما مدل را تغییر نمی‌دهیم، می‌توانیم مستقیماً از مدل اصلی استفاده کنیم اگر مطمئن هستیم.
                # برای اطمینان، از کپی استفاده می‌کنیم.
                model_bs = copy.deepcopy(base_model)
                res_bs = evaluate_zero_shot(model_bs, tokenizer, task, config, device, method=None)
                
                # --- Compressed ---
                # اینجا مدل را کپی می‌کنیم و سپس فشرده‌سازی می‌کنیم
                model_comp = copy.deepcopy(base_model)
                model_comp, stats, reduction_pct = compressor.compress_model(model_comp, method)
                res_comp = evaluate_zero_shot(model_comp, tokenizer, task, config, device, method)
                
                comp_params = sum(p.numel() for p in model_comp.parameters())
                comp_ratio = comp_params / orig_params
                
                results_summary.append({
                    'model': model_name,
                    'task': task_name,
                    'method': method,
                    'baseline_acc': res_bs['accuracy'],
                    'comp_acc': res_comp['accuracy'],
                    'comp_f1': res_comp['f1'],
                    'params_orig': orig_params,
                    'params_comp': comp_params,
                    'reduction_pct': reduction_pct,
                    'acc_drop': res_bs['accuracy'] - res_comp['accuracy']
                })
                
                # پاک کردن مدل‌های موقت برای آزادسازی RAM
                del model_bs, model_comp, stats
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n📊 Generating Comprehensive Reports...")
    
    df_results = pd.DataFrame(results_summary)
    save_path = f"{config.output_dir}/sota_final.csv"
    df_results.to_csv(save_path, index=False)
    print(f"✅ Results saved to: {save_path}")
    
    # Visualization (با تابع کمکی safe_plot_y که قبلاً تعریف کردیم)
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Accuracy vs Method (Zero-Shot)", 
        "Compression Ratio vs Accuracy Drop",
        "Performance by Model",
        "Efficiency Score (Acc / Reduction)"
    ))
    
    methods = df_results['method'].unique()
    
    def safe_plot_y(series):
        if series.empty:
            return [0.0]
        mean_val = series.mean()
        return [mean_val.item()] if hasattr(mean_val, 'item') else [float(mean_val)]

    y_vals = [safe_plot_y(df_results.loc[df_results['method'] == meth, 'comp_acc'])[0] for meth in methods]
    fig.add_trace(go.Bar(x=methods, y=y_vals, name='Avg Comp Acc', marker_color='#1f77b4'), row=1, col=1)
    
    group_by_method = df_results.groupby('method').agg({
        'reduction_pct': 'mean',
        'acc_drop': 'mean'
    }).reset_index()
    
    fig.add_trace(go.Scatter(x=group_by_method['reduction_pct'], y=group_by_method['acc_drop'], 
                             mode='markers+text', text=methods, name='Acc Drop'), row=1, col=2)
    
    models = df_results['model'].unique()
    hybrid_series = df_results[df_results['method']=='hybrid_advanced']['comp_acc']
    standard_series = df_results[df_results['method']=='standard_svd']['comp_acc']
    
    fig.add_trace(go.Bar(x=models, y=safe_plot_y(hybrid_series), name='Hybrid', marker_color='#2ca02c'), row=2, col=1)
    fig.add_trace(go.Bar(x=models, y=safe_plot_y(standard_series), name='Standard', marker_color='#d62728'), row=2, col=1)
    
    efficiency_scores = df_results.groupby('method')['comp_acc'].mean() / (df_results.groupby('method')['reduction_pct'].mean() + 1)
    fig.add_trace(go.Bar(x=methods, y=[v.item() if hasattr(v, 'item') else float(v) for v in efficiency_scores.values], 
                         name='Efficiency Score', marker_color='#ff7f0e'), row=2, col=2)
    
    fig.update_layout(height=800, title_text="Optimized SOTA Zero-Shot Lab", showlegend=True)
    fig.update_yaxes(title_text="Accuracy / Percentage")
    
    plot_path = f"{config.output_dir}/final_chart.html"
    fig.write_html(plot_path)
    print(f"✅ Chart saved to: {plot_path}")
    
    # Summary
    summary_path = f"{config.output_dir}/summary.txt"
    with open(summary_path, 'w') as f:
        f.write("SOTA ZERO-SHOT COMPRESSION SUMMARY (OPTIMIZED)\n")
        f.write("="*50 + "\n")
        f.write(f"Date: {datetime.now()}\n")
        f.write(f"Models: {list(df_results['model'].unique())}\n")
        f.write(f"Methods: {list(methods)}\n\n")
        
        f.write("BEST PERFORMING METHODS:\n")
        for meth in methods:
            sub_df = df_results[df_results['method'] == meth]
            avg_acc = sub_df['comp_acc'].mean()
            avg_red = sub_df['reduction_pct'].mean()
            avg_drop = sub_df['acc_drop'].mean()
            
            status = "✅ Excellent" if avg_drop < 0.02 else ("⚠️ Good" if avg_drop < 0.05 else "❌ High Impact")
            
            f.write(f"\nMethod: {meth}\n")
            f.write(f"  Avg Accuracy: {avg_acc:.4f}\n")
            f.write(f"  Avg Reduction: {avg_red:.2f}%\n")
            f.write(f"  Avg Drop: {avg_drop:.4f} [{status}]\n")
        
        best_row = df_results.sort_values(by=['acc_drop', 'reduction_pct']).iloc[0]
        f.write(f"\nRECOMMENDATION: Model='{best_row['model']}' + Method='{best_row['method']}'\n")
        f.write(f"Achieving {best_row['comp_acc']:.4f} acc with {best_row['reduction_pct']:.2f}% reduction.\n")
    
    print(f"✅ Summary saved to: {summary_path}")
    print("\n🎉 EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    try:
        run_comprehensive_experiment()
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        traceback.print_exc()