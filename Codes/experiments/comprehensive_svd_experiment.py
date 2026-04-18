import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from framework.utils.architecture_utils import detect_architecture, get_layer_patterns
from framework.compression.svd_compressor import SVDLinearLayer
from framework.compression.adaptive_compressor import AdaptiveTransformerCompressor

# ========== Configuration ==========
class Config:
    data_dir = "./data"
    output_dir = "./svd_comprehensive_results"
    base_models = ["distilbert-base-uncased", "roberta-base", "albert-base-v2"]
    energy_threshold = 0.95
    min_rank = 16
    batch_size = 8
    max_length = 128
    num_labels = 2
    epochs = 3
    lr = 2e-5
    enable_finetuning = True
    limit_tasks = None
    limit_samples_per_task = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

config = Config()
Path(config.output_dir).mkdir(exist_ok=True, parents=True)

# ========== Data Loading ==========
def load_datasets(data_dir, limit_tasks=None, limit_samples=1000):
    csv_files = sorted(glob.glob(f"{data_dir}/*.csv"))
    tasks = []
    for i, file_path in enumerate(tqdm(csv_files)):
        if limit_tasks and i >= limit_tasks:
            break
        try:
            df = pd.read_csv(file_path)
            # Find text and label columns
            text_col = next((c for c in df.columns if 'text' in c.lower()), None)
            label_col = next((c for c in df.columns if 'label' in c.lower()), None)
            if text_col is None or label_col is None:
                continue
            df = df.rename(columns={text_col: 'text', label_col: 'label'})
            df = df.dropna(subset=['text', 'label'])
            if len(df) > limit_samples:
                df = df.sample(n=limit_samples, random_state=42)
            split = int(len(df) * 0.8)
            tasks.append({
                'name': Path(file_path).stem,
                'train': df.iloc[:split].reset_index(drop=True),
                'val': df.iloc[split:].reset_index(drop=True)
            })
        except Exception as e:
            print(f"⚠️ Error loading {file_path}: {e}")
    return tasks

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = int(self.labels.iloc[idx])
        enc = self.tokenizer(text, truncation=True, padding='max_length',
                             max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ========== Training & Evaluation ==========
def train_and_evaluate(model, tokenizer, task_data, config, mode_name):
    train_ds = TextDataset(task_data['train']['text'], task_data['train']['label'], tokenizer, config.max_length)
    val_ds = TextDataset(task_data['val']['text'], task_data['val']['label'], tokenizer, config.max_length)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    model.to(config.device)
    if mode_name == "fine_tune":
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
        best_val_acc = 0.0
        for epoch in range(config.epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                optimizer.zero_grad()
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = F.cross_entropy(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # Validation
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(config.device)
                    attention_mask = batch['attention_mask'].to(config.device)
                    labels = batch['labels'].to(config.device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=-1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total
            best_val_acc = max(best_val_acc, val_acc)
            print(f"      Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f}")
        final_accuracy = best_val_acc
    else:
        # Zero-shot evaluation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(config.device)
                attention_mask = batch['attention_mask'].to(config.device)
                labels = batch['labels'].to(config.device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        final_accuracy = accuracy_score(all_labels, all_preds)

    # Final detailed metrics
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(config.device)
            attention_mask = batch['attention_mask'].to(config.device)
            labels = batch['labels'].to(config.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds, average='weighted'),
        'best_val_acc': final_accuracy,
        'mode': mode_name
    }

# ========== Main ==========
def run_comprehensive_experiment():
    print("="*80)
    print("🚀 STARTING COMPREHENSIVE SVD EXPERIMENT")
    print("="*80)
    tasks = load_datasets(config.data_dir, config.limit_tasks, config.limit_samples_per_task)
    if not tasks:
        print("❌ No tasks found!")
        return

    results_summary = []
    for model_name in config.base_models:
        print(f"\n{'='*60}\n🔍 Processing Model: {model_name}\n{'='*60}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.num_labels)
        base_model.to(config.device)
        arch = detect_architecture(model_name)
        patterns = get_layer_patterns(arch)
        compressor = AdaptiveTransformerCompressor(energy_threshold=config.energy_threshold, min_rank=config.min_rank)

        for task in tasks:
            task_name = task['name']
            print(f"\n   📄 Task: {task_name}")

            # Baseline Zero‑shot
            print("   [1] Baseline (Zero‑Shot)...")
            model_bs = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.num_labels)
            res_bs = train_and_evaluate(model_bs, tokenizer, task, config, mode_name="zero_shot")

            # Full Fine‑tune
            if config.enable_finetuning:
                print("   [2] Full Fine‑Tune...")
                model_ft = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.num_labels)
                res_ft = train_and_evaluate(model_ft, tokenizer, task, config, mode_name="fine_tune")
            else:
                res_ft = None

            # Compressed (Zero‑shot)
            print("   [3] Compressed (Zero‑Shot)...")
            model_comp = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.num_labels)
            model_comp, _ = compressor.compress_model(model_comp)
            res_comp = train_and_evaluate(model_comp, tokenizer, task, config, mode_name="zero_shot")

            # Compressed Fine‑tune
            if config.enable_finetuning:
                print("   [4] Compressed Fine‑Tune...")
                model_cft = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=config.num_labels)
                model_cft, _ = compressor.compress_model(model_cft)
                res_cft = train_and_evaluate(model_cft, tokenizer, task, config, mode_name="fine_tune")
            else:
                res_cft = None

            # Parameter counts
            orig_params = sum(p.numel() for p in base_model.parameters())
            comp_params = sum(p.numel() for p in model_comp.parameters())
            comp_ratio = comp_params / orig_params

            results_summary.append({
                'model': model_name,
                'task': task_name,
                'baseline_acc': res_bs['accuracy'],
                'ft_acc': res_ft['accuracy'] if res_ft else None,
                'comp_zero_acc': res_comp['accuracy'],
                'comp_ft_acc': res_cft['accuracy'] if res_cft else None,
                'compression_ratio': comp_ratio,
                'rank_used': config.min_rank
            })

    # Save results
    df = pd.DataFrame(results_summary)
    df.to_csv(f"{config.output_dir}/full_results.csv", index=False)
    print(f"\n✅ Results saved to {config.output_dir}/full_results.csv")

    # Interactive Plotly chart
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Accuracy Comparison", "F1 Score", "Compression Ratio", "Efficiency Gain"))
    models = df['model'].unique()
    mode_cols = ['baseline_acc', 'ft_acc', 'comp_zero_acc', 'comp_ft_acc']
    mode_names = ['Baseline (ZS)', 'Full FT', 'Comp (ZS)', 'Comp FT']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i, (col, name, color) in enumerate(zip(mode_cols, mode_names, colors)):
        if col == 'ft_acc' and not config.enable_finetuning:
            continue
        y_vals = [df[df['model']==m][col].mean() for m in models]
        fig.add_trace(go.Bar(x=models, y=y_vals, name=name, marker_color=color), row=1, col=(i//2)+1)
    fig.add_trace(go.Bar(x=models, y=df.groupby('model')['compression_ratio'].mean(), name="Avg Comp Ratio", marker_color='#9467bd'), row=2, col=1)
    fig.update_layout(height=800, title_text="Comprehensive SVD Performance Analysis")
    fig.write_html(f"{config.output_dir}/performance_analysis.html")
    print(f"✅ Interactive plot saved to {config.output_dir}/performance_analysis.html")

    # Text summary
    with open(f"{config.output_dir}/summary.txt", 'w') as f:
        f.write("EXPERIMENT SUMMARY\n" + "="*50 + "\n")
        f.write(f"Models: {', '.join(models)}\n")
        f.write(f"Tasks: {df['task'].nunique()}\n")
        f.write(f"Energy threshold: {config.energy_threshold}\n")
        f.write(f"Min rank: {config.min_rank}\n\n")
        for m in models:
            sub = df[df['model']==m]
            f.write(f"{m}:\n")
            f.write(f"  Avg FT Accuracy: {sub['ft_acc'].mean():.4f}\n")
            f.write(f"  Avg Comp FT Accuracy: {sub['comp_ft_acc'].mean():.4f}\n")
            f.write(f"  Avg Compression Ratio: {sub['compression_ratio'].mean():.2%}\n\n")
    print(f"✅ Summary saved to {config.output_dir}/summary.txt")
    print("\n🎉 EXPERIMENT COMPLETE!")

if __name__ == "__main__":
    run_comprehensive_experiment()