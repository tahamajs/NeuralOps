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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm import tqdm

warnings.filterwarnings("ignore")
torch.set_grad_enabled(True)
path = "/Users/tahamajs/Documents/uni/Project/models_cache/hub"
@dataclass
class ExperimentConfig:
    data_dir: str = "./data"
    output_dir: str = "./svd_comprehensive_results"
    
    base_models: List[str] = field(default_factory=lambda: [
        "distilbert-base-uncased",
        "roberta-base",
        "albert-base-v2"
    ])
    
    energy_threshold: float = 0.95  # درصد انرژی نگه داشته شده
    min_rank: int = 16              # حداقل رتبه مجاز
    
    batch_size: int = 8
    max_length: int = 128
    num_labels: int = 2             # فرض بر دو کلاسه بودن (برای متنوع بودن داده‌ها قابل تغییر است)
    epochs: int = 3                 # تعداد اپوک برای فاین‌تیونینگ
    lr: float = 2e-5                # نرخ یادگیری
    
    enable_finetuning: bool = True
    limit_tasks: int = None         # اگر عدد باشد فقط آن تعداد تسک اول را تست می‌کند
    limit_samples_per_task: int = 1000 # محدودیت نمونه برای سرعت بیشتر در تست
    
    def __post_init__(self):
        Path(self.output_dir).mkdir(exist_ok=True, parents=True)

config = ExperimentConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

print(f"🚀 Running on device: {device}")

# ==================== 1. COMPRESSION MODULE (ATTENTION SPECIFIC) ====================

class CompressedAttentionLayer(nn.Module):
    """
    لایه Attention فشرده شده با استفاده از SVD روی وزن‌های Q, K, V, O.
    این کلاس وزن‌های اصلی را تجزیه کرده و ساختار را به دو مرحله ضرب ماتریسی کوچکتر تبدیل می‌کند.
    """
    def __init__(self, original_layer: nn.MultiheadAttention, energy_threshold: float = 0.95, min_rank: int = 16):
        super().__init__()
        
        self.original_dim = original_layer.embed_dim
        self.num_heads = original_layer.num_heads
        self.head_dim = self.original_dim // self.num_heads
        
        # استخراج وزن‌های اصلی
        w_q = original_layer.in_proj_weight.data.cpu().numpy() 
        w_o = original_layer.out_proj.weight.data.cpu().numpy() 
        
        # 1. تجزیه in_proj_weight (Q, K, V)
        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_q, full_matrices=False)
        energy_qkv = np.cumsum(S_qkv) / np.sum(S_qkv)
        rank_qkv = np.argmax(energy_qkv >= energy_threshold) + 1
        rank_qkv = max(min_rank, min(rank_qkv, w_q.shape[1]))
        
        # 2. تجزیه out_proj_weight (O)
        U_out, S_out, Vh_out = np.linalg.svd(w_o, full_matrices=False)
        energy_out = np.cumsum(S_out) / np.sum(S_out)
        rank_out = np.argmax(energy_out >= energy_threshold) + 1
        rank_out = max(min_rank, min(rank_out, w_o.shape[1]))
        
        self.rank_qkv = rank_qkv
        self.rank_out = rank_out
        
        # ذخیره پارامترهای فشرده شده (Buffers برای عدم محاسبه گرادیان در زمان فشرده‌سازی اولیه)
        # اما در فاین‌تیونینگ باید trainable باشند. اینجا آنها را Parameter می‌کنیم تا قابلیت آپدیت داشته باشند.
        self.register_parameter('U_qkv', nn.Parameter(torch.from_numpy(U_qkv[:, :rank_qkv]).float()))
        self.register_parameter('S_qkv_sqrt', nn.Parameter(torch.from_numpy(np.sqrt(S_qkv[:rank_qkv])).float()))
        self.register_parameter('Vh_qkv', nn.Parameter(torch.from_numpy(Vh_qkv[:rank_qkv, :]).float()))
        
        self.register_parameter('U_out', nn.Parameter(torch.from_numpy(U_out[:, :rank_out]).float()))
        self.register_parameter('S_out_sqrt', nn.Parameter(torch.from_numpy(np.sqrt(S_out[:rank_out])).float()))
        self.register_parameter('Vh_out', nn.Parameter(torch.from_numpy(Vh_out[:rank_out, :]).float()))
        
        # بیاسها
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

    def forward(self, query, key, value, key_padding_mask=None, 
                need_weights=True, attn_mask=None, average_attn_weights=True):
        x = query
        dim = self.original_dim
        
        # 1. تولید Q, K, V فشرده
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
        
        # 2. خروجی نهایی فشرده
        context_fused = torch.matmul(context, self.Vh_out.t() * self.S_out_sqrt.view(1, -1, 1))
        final_output = torch.matmul(context_fused, self.U_out.t())
        
        if self.bias_out is not None:
            final_output += self.bias_out
            
        return final_output, attn_weights if need_weights else None


class AdaptiveTransformerCompressor:
    def __init__(self, energy_threshold: float = 0.95, min_rank: int = 16):
        self.energy_threshold = energy_threshold
        self.min_rank = min_rank
        
    def compress_model(self, model: nn.Module) -> nn.Module:
        print(f"🔧 Starting Compression (Threshold: {self.energy_threshold}, Min Rank: {self.min_rank})")
        compression_stats = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                try:
                    compressed_layer = CompressedAttentionLayer(
                        module, 
                        energy_threshold=self.energy_threshold, 
                        min_rank=self.min_rank
                    )
                    
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    
                    if parent_name:
                        parent = model.get_submodule(parent_name)
                    else:
                        parent = model
                    
                    setattr(parent, child_name, compressed_layer)
                    
                    orig_params = sum(p.numel() for p in module.parameters())
                    new_params = sum(p.numel() for p in compressed_layer.parameters())
                    
                    compression_stats.append({
                        'layer': name,
                        'original_params': orig_params,
                        'compressed_params': new_params,
                        'ratio': new_params / orig_params
                    })
                    
                except Exception as e:
                    print(f"   ❌ Failed to compress {name}: {e}")
        
        orig_total = sum(p.numel() for p in model.parameters())
        new_total = sum(p.numel() for p in model.parameters())
        total_ratio = new_total / orig_total
        
        print(f"✅ Compression Complete! Total Params: {orig_total} -> {new_total} (Ratio: {total_ratio:.2%})")
        return model, compression_stats

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
            # تشخیص ستون‌ها
            text_cols = [c for c in df.columns if 'text' in c.lower()]
            label_cols = [c for c in df.columns if 'label' in c.lower()]
            
            if not text_cols or not label_cols:
                continue
                
            text_col = text_cols[0]
            label_col = label_cols[0]
            
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            df = df.dropna(subset=['text', 'label'])
            
            # محدودیت نمونه‌ها
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

# ==================== 3. TRAINING & EVALUATION ====================

def train_and_evaluate(model, tokenizer, task_data, config, device, mode_name: str):
    """
    تابع اصلی برای آموزش یا ارزیابی سریع.
    """
    train_ds = TextDataset(task_data['train']['text'], task_data['train']['label'], tokenizer, config.max_length)
    val_ds = TextDataset(task_data['val']['text'], task_data['val']['label'], tokenizer, config.max_length)
    
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    best_val_acc = 0.0
    
    if mode_name == "fine_tune":
        print(f"   🎓 Fine-tuning for {config.epochs} epochs...")
        for epoch in range(config.epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = F.cross_entropy(logits, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            val_acc = val_correct / val_total
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            model.train()
            print(f"      Epoch {epoch+1} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")
            
    else:
        # Zero-shot evaluation (بدون آموزش)
        print(f"   ⚡ Evaluating Zero-Shot...")
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        best_val_acc = val_correct / val_total
        print(f"      Zero-Shot Val Acc: {best_val_acc:.4f}")

    # Final Metrics
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
    
    return {
        'accuracy': acc,
        'f1': f1,
        'best_val_acc': best_val_acc,
        'mode': mode_name
    }

# ==================== 4. MAIN PIPELINE ====================

# ==================== 4. MAIN PIPELINE ====================

def run_comprehensive_experiment():
    print("\n" + "="*80)
    print("🚀 STARTING COMPREHENSIVE SVD EXPERIMENT")
    print("="*80)
    
    # 1. Load Data
    tasks = load_datasets(config.data_dir, limit_tasks=config.limit_tasks, limit_samples=config.limit_samples_per_task)
    if not tasks:
        print("❌ No tasks found!")
        return

    results_summary = []
    for model_name in config.base_models:
        print(f"\n{'='*60}")
        print(f"🔍 Processing Model: {model_name}")
        print(f"{'='*60}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        base_model = None
        try:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=config.num_labels
            )
        except Exception as e:
            print(f"❌ Failed to load {model_name}: {e}")
            continue 
        
        if base_model is None:
            print(f"⚠️ Warning: base_model is still None after loading attempt.")
            continue

        for task in tasks:
            task_name = task['name']
            print(f"\n   📄 Task: {task_name}")
            
            print("   [1] Baseline (Zero-Shot)...")
            model_bs = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=config.num_labels
            )
            res_bs = train_and_evaluate(model_bs, tokenizer, task, config, device, mode_name="zero_shot")
            
            # --- حالت 2: Full Fine-Tune ---
            if config.enable_finetuning:
                print("   [2] Full Fine-Tune...")
                model_ft = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=config.num_labels
                )
                res_ft = train_and_evaluate(model_ft, tokenizer, task, config, device, mode_name="fine_tune")
            else:
                res_ft = None

            print("   [3] Compressed (Zero-Shot)...")
            model_comp = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=config.num_labels
            )
            compressor = AdaptiveTransformerCompressor(energy_threshold=config.energy_threshold, min_rank=config.min_rank)
            model_comp, stats = compressor.compress_model(model_comp)
            res_comp = train_and_evaluate(model_comp, tokenizer, task, config, device, mode_name="zero_shot")
            
            if config.enable_finetuning:
                print("   [4] Compressed Fine-Tune...")
                model_cft = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=config.num_labels
                )
                model_cft, _ = compressor.compress_model(model_cft)
                res_cft = train_and_evaluate(model_cft, tokenizer, task, config, device, mode_name="fine_tune")
            else:
                res_cft = None
            
            if base_model is not None:
                total_params_orig = sum(p.numel() for p in base_model.parameters())
                total_params_comp = sum(p.numel() for p in model_comp.parameters())
                comp_ratio = total_params_comp 

    print("\n📊 Generating Reports and Visualizations...")
    
    df_results = pd.DataFrame(results_summary)
    
    save_path = f"{config.output_dir}/full_results.csv"
    df_results.to_csv(save_path, index=False)
    print(f"✅ Results saved to: {save_path}")
    
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Accuracy Comparison", "F1 Score Comparison", "Compression Ratio", "Efficiency Gain"))
    
    models = df_results['model'].unique()
    modes = ['baseline_acc', 'ft_acc', 'comp_zero_acc', 'comp_ft_acc']
    mode_names = ['Baseline (ZS)', 'Full FT', 'Comp (ZS)', 'Comp FT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, mode in enumerate(modes):
        if mode == 'ft_acc' and config.enable_finetuning == False: continue
        if mode == 'comp_ft_acc' and config.enable_finetuning == False: continue
        
        y_vals = []
        for m in models:
            mask = df_results['model'] == m
            vals = df_results.loc[mask, mode].dropna()
            if len(vals) > 0:
                y_vals.append(vals.mean())
            else:
                y_vals.append(None)
        
        fig.add_trace(go.Bar(x=models, y=y_vals, name=mode_names[i], marker_color=colors[i]), row=1, col=(i//2)+1)
    
    fig.add_trace(go.Bar(x=models, y=df_results.groupby('model')['compression_ratio'].mean(), name="Avg Comp Ratio", marker_color='#9467bd'), row=2, col=1)
    
    fig.update_layout(height=800, title_text="Comprehensive SVD Performance Analysis", showlegend=True)
    fig.update_yaxes(title_text="Accuracy / F1 Score")
    
    plot_path = f"{config.output_dir}/performance_analysis.html"
    fig.write_html(plot_path)
    print(f"✅ Interactive Plot saved to: {plot_path}")
    
    summary_path = f"{config.output_dir}/summary.txt"
    with open(summary_path, 'w') as f:
        f.write("EXPERIMENT SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Total Models Tested: {len(models)}\n")
        f.write(f"Total Tasks Tested: {df_results['task'].nunique()}\n")
        f.write(f"Energy Threshold: {config.energy_threshold}\n")
        f.write(f"Min Rank: {config.min_rank}\n\n")
        
        f.write("AVERAGE RESULTS BY MODEL:\n")
        for m in models:
            sub_df = df_results[df_results['model'] == m]
            f.write(f"\n{m}:\n")
            f.write(f"  Avg Accuracy (FT): {sub_df['ft_acc'].mean():.4f}\n")
            f.write(f"  Avg Accuracy (Comp FT): {sub_df['comp_ft_acc'].mean():.4f}\n")
            f.write(f"  Avg Compression Ratio: {sub_df['compression_ratio'].mean():.2%}\n")
            
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