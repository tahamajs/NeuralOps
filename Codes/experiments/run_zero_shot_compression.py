#!/usr/bin/env python3
import sys, os, gc, json, copy
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework.compression.methods import AdvancedZeroShotCompressor
from framework.data.custom_loader import load_datasets
from framework.utils.device import get_device
from framework.config.default_config import config

def evaluate_zero_shot(model, tokenizer, val_loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    return {'accuracy': accuracy_score(labels, preds), 'f1': f1_score(labels, preds, average='weighted')}

def main():
    print("🚀 Starting Zero‑Shot Compression Lab (Optimised)")
    device = get_device()
    tasks = load_datasets(config.data_dir, limit_tasks=config.limit_tasks, limit_samples=config.limit_samples_per_task)
    if not tasks:
        print("No tasks found.")
        return

    results = []
    for model_name in config.base_models:
        print(f"\n{'#'*60}\nModel: {model_name}\n{'#'*60}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        base_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True).to(device)
        orig_params = sum(p.numel() for p in base_model.parameters())

        for method in config.compression_methods:
            print(f"\n>>> Method: {method}")
            compressor = AdvancedZeroShotCompressor(method, config.energy_threshold_base, config.min_rank_global)
            for task in tasks:
                print(f"   Task: {task['name']}")
                # Prepare validation loader
                from framework.data.dataset_utils import TextDataset
                val_ds = TextDataset(task['val']['text'], task['val']['label'], tokenizer, config.max_length)
                val_loader = DataLoader(val_ds, batch_size=config.batch_size)

                # Baseline (copy of original)
                model_bs = copy.deepcopy(base_model)
                res_bs = evaluate_zero_shot(model_bs, tokenizer, val_loader, device)

                # Compressed
                model_cp = copy.deepcopy(base_model)
                model_cp, stats, reduction = compressor.compress_model(model_cp, method)
                res_cp = evaluate_zero_shot(model_cp, tokenizer, val_loader, device)

                results.append({
                    'model': model_name,
                    'task': task['name'],
                    'method': method,
                    'baseline_acc': res_bs['accuracy'],
                    'comp_acc': res_cp['accuracy'],
                    'comp_f1': res_cp['f1'],
                    'reduction_pct': reduction,
                    'acc_drop': res_bs['accuracy'] - res_cp['accuracy']
                })
                del model_bs, model_cp
                gc.collect()

    df = pd.DataFrame(results)
    df.to_csv(f"{config.output_dir}/zero_shot_results.csv", index=False)
    print("✅ Results saved.")

    # Plotting
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Accuracy by Method", "Accuracy Drop vs Reduction", "Per‑Model Performance", "Efficiency"))
    methods = df['method'].unique()
    acc_by_method = [df[df['method']==m]['comp_acc'].mean() for m in methods]
    fig.add_trace(go.Bar(x=methods, y=acc_by_method, name='Avg Acc'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.groupby('method')['reduction_pct'].mean(), y=df.groupby('method')['acc_drop'].mean(), mode='markers+text', text=methods), row=1, col=2)
    # etc. – full plotting as in original script
    fig.write_html(f"{config.output_dir}/zero_shot_chart.html")
    print("✅ Chart saved.")

if __name__ == "__main__":
    main()