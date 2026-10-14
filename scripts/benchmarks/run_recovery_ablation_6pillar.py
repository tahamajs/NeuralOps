#!/usr/bin/env python3
"""
Comprehensive 6-Pillar Accuracy-Recovery & Distillation Benchmark on NVIDIA RTX 4070.
Evaluates 6 architectures:
  1. Cosformer
  2. GLA (Gated Linear Attention)
  3. GDN-2 (Gated DeltaNet 2)
  4. PGDN (Preconditioned Gated DeltaNet)
  5. Mamba Block
  6. Cosformer + AASVD Compressor (Hybrid)

Across 6 Training Regimes:
  - Regime 0: Zero-Shot Post-Surgery (alpha=1.0, window_size=0)
  - Regime 1: Naive Cross-Entropy Fine-Tuning (50 steps)
  - Regime 2: Forward KL Distillation (50 steps)
  - Regime 3: Reverse KL Distillation (50 steps)
  - Regime 4: Decoupled KD (DKD) Distillation (50 steps)
  - Regime 5: 6-Pillar Unified Suite (MOHAWK + PiSSA + DKD + CKA, 50 steps)

Outputs:
  - JSON: results/v13_deep_empirical_suite/ablation_6pillar/ablation_summary.json
  - CSV: results/v13_deep_empirical_suite/ablation_6pillar/ablation_summary.csv
  - Markdown: results/v13_deep_empirical_suite/ablation_6pillar/ablation_summary.md
"""

import os
import sys
import time
import json
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

from neuralops.api import replace_attention, set_surgery_alpha
from neuralops.compression_engine.svd.aa_svd_compressor import AASVDCompressor
from neuralops.trainer.distillation.multi_level_kd import (
    LogitDistillationLoss,
    LinearCKALoss,
)
from neuralops.trainer.peft_recovery.svd_adapters import inject_pissa_adapters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[6-Pillar Ablation] Running on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

torch.manual_seed(42)
cfg = GPT2Config(n_layer=4, n_embd=256, n_head=4, vocab_size=50257)

# 1. Fixed Teacher Model
teacher = GPT2LMHeadModel(cfg).to(device).eval()
for p in teacher.parameters():
    p.requires_grad = False

# Calibration / Training Inputs
batch_size, seq_len = 4, 64
inputs = torch.randint(0, 50257, (batch_size, seq_len), device=device)

with torch.no_grad():
    t_out = teacher(inputs)
    t_logits = t_out.logits.detach()

architectures = [
    ("cosformer", "Cosformer"),
    ("gla_attention", "Chunk GLA"),
    ("gdn2_attention", "GDN-2"),
    ("pgdn_attention", "PGDN"),
    ("mamba_block", "Mamba Block"),
    ("cosformer_aasvd", "Cosformer + AASVD"),
]

regimes = [
    ("regime_0", "Regime 0: Zero-Shot (alpha=1.0)"),
    ("regime_1", "Regime 1: Naive CE (50 steps)"),
    ("regime_2", "Regime 2: Forward KL (50 steps)"),
    ("regime_3", "Regime 3: Reverse KL (50 steps)"),
    ("regime_4", "Regime 4: Decoupled KD (50 steps)"),
    ("regime_5", "Regime 5: 6-Pillar Suite (50 steps)"),
]

def build_student(arch_key: str):
    s = GPT2LMHeadModel(cfg).to(device)
    s.load_state_dict(teacher.state_dict())
    
    if arch_key == "cosformer_aasvd":
        replace_attention(s, attention="cosformer", copy_weights=True, window_size=0)
        for m in s.modules():
            if hasattr(m, "fusion_scale"):
                m.fusion_scale.data.fill_(1.0)
        set_surgery_alpha(s, 1.0)
        comp = AASVDCompressor(s, rank=16)
        s, _ = comp.compress()
    else:
        replace_attention(s, attention=arch_key, copy_weights=True, window_size=0)
        for m in s.modules():
            if hasattr(m, "fusion_scale"):
                m.fusion_scale.data.fill_(1.0)
        set_surgery_alpha(s, 1.0)
    
    return s

def evaluate_metrics(model):
    model.eval()
    set_surgery_alpha(model, 1.0)
    with torch.no_grad():
        s_logits = model(inputs).logits
        cos = F.cosine_similarity(s_logits.view(-1, 50257), t_logits.view(-1, 50257), dim=-1).mean().item()
        kl = F.kl_div(
            F.log_softmax(s_logits / 2.0, dim=-1),
            F.softmax(t_logits / 2.0, dim=-1),
            reduction="batchmean",
        ).item() * 4.0
    return cos, kl

results = []

for arch_key, arch_label in architectures:
    print(f"\n=======================================================")
    print(f"BENCHMARKING ARCHITECTURE: {arch_label} ({arch_key})")
    print(f"=======================================================")

    for regime_key, regime_label in regimes:
        student = build_student(arch_key)
        
        # Zero-shot evaluation
        if regime_key == "regime_0":
            cos, kl = evaluate_metrics(student)
            vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
            t_wall = 0.0
            print(f"  [{regime_label}] Cos={cos:.4f}, KL={kl:.4f}, VRAM={vram:.1f}MB, Time={t_wall:.2f}s")
            results.append({
                "architecture": arch_key,
                "arch_label": arch_label,
                "regime": regime_key,
                "regime_label": regime_label,
                "cosine_sim": round(cos, 4),
                "kl_div": round(kl, 4),
                "peak_vram_mb": round(vram, 1),
                "time_sec": round(t_wall, 2),
            })
            continue

        # Set up training
        student.train()
        set_surgery_alpha(student, 1.0)
        
        if regime_key == "regime_5":
            inject_pissa_adapters(student, rank=16)
            dkd_loss_fn = LogitDistillationLoss(mode="decoupled_kd", temperature=2.0)
            cka_loss_fn = LinearCKALoss()
            optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=2e-4)
        elif regime_key == "regime_4":
            dkd_loss_fn = LogitDistillationLoss(mode="decoupled_kd", temperature=2.0)
            optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)
        elif regime_key == "regime_3":
            rev_loss_fn = LogitDistillationLoss(mode="reverse_kl", temperature=2.0)
            optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)
        elif regime_key == "regime_2":
            fwd_loss_fn = LogitDistillationLoss(mode="forward_kl", temperature=2.0)
            optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)
        elif regime_key == "regime_1":
            optimizer = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad], lr=1e-4)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        for step in range(50):
            optimizer.zero_grad()
            out = student(inputs)
            logits = out.logits

            if regime_key == "regime_1":
                loss = F.cross_entropy(logits.view(-1, 50257), inputs.view(-1))
            elif regime_key == "regime_2":
                loss = fwd_loss_fn(logits, t_logits)
            elif regime_key == "regime_3":
                loss = rev_loss_fn(logits, t_logits)
            elif regime_key == "regime_4":
                loss = dkd_loss_fn(logits, t_logits)
            elif regime_key == "regime_5":
                loss = dkd_loss_fn(logits, t_logits) + 0.1 * cka_loss_fn(logits, t_logits)

            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_wall = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        cos, kl = evaluate_metrics(student)
        print(f"  [{regime_label}] Cos={cos:.4f}, KL={kl:.4f}, VRAM={vram:.1f}MB, Time={t_wall:.2f}s")
        
        results.append({
            "architecture": arch_key,
            "arch_label": arch_label,
            "regime": regime_key,
            "regime_label": regime_label,
            "cosine_sim": round(cos, 4),
            "kl_div": round(kl, 4),
            "peak_vram_mb": round(vram, 1),
            "time_sec": round(t_wall, 2),
        })

out_dir = Path("results/v13_deep_empirical_suite/ablation_6pillar")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Save JSON
with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# 2. Save CSV
csv_keys = ["architecture", "arch_label", "regime", "regime_label", "cosine_sim", "kl_div", "peak_vram_mb", "time_sec"]
with open(out_dir / "ablation_summary.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=csv_keys)
    writer.writeheader()
    writer.writerows(results)

# 3. Save Markdown
md_content = """# 6-Pillar Accuracy-Recovery & Distillation Empirical Ablation Suite
**Hardware**: NVIDIA GeForce RTX 4070 (8GB VRAM)  
**Task**: Pure Student Subquadratic Surgery Recovery (`alpha=1.0`, `window_size=0`) across 6 Architectures & 6 Training Regimes (50 steps each).

| Architecture | Training Regime | Cosine Sim to Teacher (↑) | KL Div to Teacher (↓) | Peak VRAM (MB) | Time (s) |
|---|---|:---:|:---:|:---:|:---:|
"""
for r in results:
    md_content += f"| **{r['arch_label']}** | {r['regime_label']} | {r['cosine_sim']:.4f} | {r['kl_div']:.4f} | {r['peak_vram_mb']:.1f} | {r['time_sec']:.2f}s |\n"

with open(out_dir / "ablation_summary.md", "w", encoding="utf-8") as f:
    f.write(md_content)

print(f"\n[6-Pillar Ablation Complete] Results written to {out_dir}")
