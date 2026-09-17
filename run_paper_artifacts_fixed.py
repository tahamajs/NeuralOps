import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import time

MODULES = ["svd_linear"]
RANKS = [16, 32, 64, 128, 192, 256, 384]
LORA_RANKS = [8, 16, 32, 64]
RECOVERY_STEPS = [10, 25, 50, 100, 200]
SEEDS = [42, 123, 456]

BACKBONES = {
    "distilbert": {"family": "bert", "batch": 32, "model_path": "models/distilbert_sst2"},
    "roberta":    {"family": "roberta", "batch": 16, "model_path": "models/roberta-base"},
    "gpt2":       {"family": "gpt", "batch": 16, "model_path": "models/gpt2"},
}

RANK_ARG = {
    "svd_linear": "rank",
    "svd_llm_v2": "rank",
}

def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)

def find_checkpoint(backbone):
    candidates = [
        f"results/distilbert_agnews_baseline/finetuned_model.pt" if backbone == "distilbert" else None,
        f"results/{backbone}_agnews_baseline/finetuned_model.pt",
        f"results/unified_asvd/{backbone}/finetuned_model.pt",
        f"results/unified_svd/{backbone}/finetuned_model.pt",
        f"results/unified_svd_linear/{backbone}/finetuned_model.pt",
        f"results/unified_fms/{backbone}/finetuned_model.pt",
        f"results/unified_galerkin/{backbone}/finetuned_model.pt",
        f"results/unified_wgqa_rerun/{backbone}/finetuned_model.pt",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None

def run_one(module, backbone, info, rank, lora_rank, rec_steps, seed, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if (Path(out_dir) / "summary.csv").exists():
        print(f"[CACHE] Skipping {out_dir}", flush=True)
        return
        
    ckpt = find_checkpoint(backbone)
    cmd = [
        sys.executable, "-u", "Codes/framework/run_quality_sweep.py",
        "--model", info["model_path"],
        "--model_family", info["family"],
        "--dataset", "ag_news",
        "--dataset_dir", "Codes/framework/data/glue_data/ag_news",
        "--output_dir", out_dir,
        "--modules", f"full,{module}",
        "--batch_size", str(info["batch"]),
        "--recovery_steps", str(rec_steps),
        "--learning_rate", "2e-4",
        "--lora_rank", str(lora_rank),
        "--lora_alpha", "64",
        f"--{RANK_ARG.get(module, 'rank')}", str(rank),
        "--precision", "fp16",
        "--fresh",
        "--defer_export",
        "--timeout", "900",
        "--seed", str(seed)
    ]
    if ckpt:
        cmd.extend(["--parent_checkpoint", ckpt])
        
    run_cmd(cmd)

def main():
    print("Starting Paper Artifacts Suite...", flush=True)
    for module in MODULES:
        print(f"=== Module: {module} ===", flush=True)
        # Task A1: Rank Ablation
        for rank in RANKS:
            for seed in SEEDS:
                for backbone, info in BACKBONES.items():
                    out_dir = f"papers/modules/{module}/runs/rank_ablation/rank_{rank}/seed_{seed}/{backbone}"
                    run_one(module, backbone, info, rank, 32, 100, seed, out_dir)
                    
        # Task A2: LoRA Rank Ablation
        for lora in LORA_RANKS:
            for backbone, info in BACKBONES.items():
                out_dir = f"papers/modules/{module}/runs/lora_ablation/lora_{lora}/{backbone}"
                run_one(module, backbone, info, 64, lora, 100, 42, out_dir)
                
        # Task A3: Recovery Steps Ablation
        for steps in RECOVERY_STEPS:
            for backbone, info in BACKBONES.items():
                out_dir = f"papers/modules/{module}/runs/rec_ablation/steps_{steps}/{backbone}"
                run_one(module, backbone, info, 64, 16, steps, 42, out_dir)

if __name__ == "__main__":
    main()
