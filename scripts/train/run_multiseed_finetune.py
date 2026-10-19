#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Run multiseed finetune module.
"""

import os
import time
import torch
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from neuralops.utils.replace_attention import replace_attention

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {device} ({torch.cuda.get_device_name(0)})")

seeds = [42, 1337, 2026]
tasks = ["mrpc", "sst2"]
num_epochs = 5
learning_rate = 3e-5
batch_size = 16

tok = AutoTokenizer.from_pretrained("/mnt/d/models/roberta-base")

benchmark_records = []

for task in tasks:
    print(f"\n==================== TASK: GLUE-{task.upper()} ====================")
    try:
        ds_train = load_dataset("nyu-mll/glue", task, split="train")
        ds_val = load_dataset("nyu-mll/glue", task, split="validation")
    except Exception:
        ds_train = load_dataset("glue", task, split="train")
        ds_val = load_dataset("glue", task, split="validation")

    train_size = 400
    val_size = 150
    sub_train = ds_train.select(range(min(len(ds_train), train_size)))
    sub_val = ds_val.select(range(min(len(ds_val), val_size)))

    def collate_fn(batch):
        """Execute collate fn routine."""
        if task == "mrpc":
            s1 = [item["sentence1"] for item in batch]
            s2 = [item["sentence2"] for item in batch]
            enc = tok(s1, s2, padding=True, truncation=True, max_length=128, return_tensors="pt")
        else:
            s = [item["sentence"] for item in batch]
            enc = tok(s, padding=True, truncation=True, max_length=128, return_tensors="pt")
        enc["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {k: v.to(device) for k, v in enc.items()}

    val_loader = DataLoader(sub_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    for seed in seeds:
        print(f"\n>>> Seed {seed} | Task {task.upper()} <<<")
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        train_loader = DataLoader(sub_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        m = AutoModelForSequenceClassification.from_pretrained("/mnt/d/models/roberta-base").to(device)
        m = replace_attention(m, attention_key="svd_linear_attention")

        optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=0.01)

        for epoch in range(1, num_epochs + 1):
            m.train()
            total_train_loss = 0.0
            num_batches = 0

            t0 = time.perf_counter()
            for batch in train_loader:
                optimizer.zero_grad()
                out = m(**batch)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                optimizer.step()
                total_train_loss += loss.item()
                num_batches += 1

            train_loss = total_train_loss / num_batches

            m.eval()
            total_val_loss = 0.0
            val_batches = 0
            correct = 0
            total_items = 0

            with torch.no_grad():
                for batch in val_loader:
                    out = m(**batch)
                    total_val_loss += out.loss.item()
                    val_batches += 1
                    preds = torch.argmax(out.logits, dim=-1)
                    correct += (preds == batch["labels"]).sum().item()
                    total_items += batch["labels"].size(0)

            val_loss = total_val_loss / val_batches
            val_acc = correct / total_items * 100.0
            epoch_time = time.perf_counter() - t0
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)

            print(
                f"Epoch {epoch}/5 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:5.2f}% | Time: {epoch_time:4.2f}s | VRAM: {peak_vram:.1f}MB"
            )

            benchmark_records.append(
                {
                    "Task": f"GLUE-{task.upper()}",
                    "Seed": seed,
                    "Epoch": epoch,
                    "Train_Loss": round(train_loss, 4),
                    "Val_Loss": round(val_loss, 4),
                    "Val_Accuracy_Pct": round(val_acc, 2),
                    "Epoch_Time_Sec": round(epoch_time, 2),
                    "Peak_VRAM_MB": round(peak_vram, 2),
                }
            )

df = pd.DataFrame(benchmark_records)
os.makedirs("/mnt/d/SVD_linear_Attention/docs/benchmarks", exist_ok=True)
csv_path = "/mnt/d/SVD_linear_Attention/docs/benchmarks/paper_benchmarks_multiseed_finetune.csv"
df.to_csv(csv_path, index=False)
print(f"\nExported {csv_path} successfully.")

summary = (
    df.groupby(["Task", "Epoch"])
    .agg({"Train_Loss": ["mean", "std"], "Val_Loss": ["mean", "std"], "Val_Accuracy_Pct": ["mean", "std"]})
    .reset_index()
)

summary.columns = [
    "Task",
    "Epoch",
    "Train_Loss_Mean",
    "Train_Loss_Std",
    "Val_Loss_Mean",
    "Val_Loss_Std",
    "Val_Acc_Mean",
    "Val_Acc_Std",
]

latex_content = r"""% Publication Benchmark Deliverable: Multi-Seed SVD Linear Attention Fine-Tuning Convergence
% Seeds: 42, 1337, 2026 | Hardware: NVIDIA GeForce RTX 4070 Laptop GPU (8GB VRAM)

\begin{table*}[t]
\centering
\small
\caption{\textbf{Multi-Seed Fine-Tuning Convergence Curves for Post-Surgery SVD Linear Attention.} Training and validation trajectory on GLUE MRPC and SST-2 across 5 epochs under seeds $\{42, 1337, 2026\}$ on physical RTX 4070 Laptop GPU. Values reported as Mean $\pm$ Standard Deviation.}
\label{tab:multiseed_finetune}
\begin{tabular}{llcccc}
\toprule
\textbf{Downstream Task} & \textbf{Epoch} & \textbf{Training Loss} $\downarrow$ & \textbf{Validation Loss} $\downarrow$ & \textbf{Validation Accuracy (\%)} $\uparrow$ \\
\midrule
"""

for _, row in summary.iterrows():
    task_name = row["Task"]
    epoch_num = int(row["Epoch"])
    tr_m, tr_s = row["Train_Loss_Mean"], (row["Train_Loss_Std"] if not pd.isna(row["Train_Loss_Std"]) else 0.0)
    vl_m, vl_s = row["Val_Loss_Mean"], (row["Val_Loss_Std"] if not pd.isna(row["Val_Loss_Std"]) else 0.0)
    acc_m, acc_s = row["Val_Acc_Mean"], (row["Val_Acc_Std"] if not pd.isna(row["Val_Acc_Std"]) else 0.0)

    latex_content += f"{task_name} & Epoch {epoch_num} & {tr_m:.4f} $\\pm$ {tr_s:.4f} & {vl_m:.4f} $\\pm$ {vl_s:.4f} & {acc_m:.2f}\\% $\\pm$ {acc_s:.2f}\\% \\\\\n"
    if epoch_num == 5 and task_name == "GLUE-MRPC":
        latex_content += "\\midrule\n"

latex_content += r"""\bottomrule
\end{tabular}
\end{table*}
"""

tex_path = "/mnt/d/SVD_linear_Attention/docs/reports/PAPER_TABLES_MULTISEED_FINETUNE.tex"
with open(tex_path, "w") as f:
    f.write(latex_content)
print(f"Exported {tex_path} successfully.")
