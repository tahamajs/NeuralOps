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
    "distilbert": {"family": "bert", "batch": 32},
    "roberta":    {"family": "roberta", "batch": 16},
    "gpt2":       {"family": "gpt", "batch": 16},
}
RANK_ARG = {
    "svd_linear": "rank",
    "svd_llm_v2": "rank",
}

def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)

def run_one(module, backbone, family, batch, rank, lora_rank, rec_steps, seed, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if (Path(out_dir) / "summary.csv").exists():
        print(f"[CACHE] Skipping {out_dir}", flush=True)
        return
        
    cmd = [
        sys.executable, "-u", "Codes/framework/run_quality_sweep.py",
        "--model", f"models/{backbone}_sst2",
        "--model_family", family,
        "--dataset", "ag_news",
        "--dataset_dir", "Codes/framework/data/glue_data/ag_news",
        "--output_dir", out_dir,
        "--parent_checkpoint", f"results/unified_asvd/{backbone}/finetuned_model.pt",
        "--modules", f"full,{module}",
        "--batch_size", str(batch),
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
    run_cmd(cmd)

def main():
    print("Starting Paper Artifacts Suite...")
    for module in MODULES:
        print(f"=== Module: {module} ===")
        # Task A1: Rank Ablation
        for rank in RANKS:
            for seed in SEEDS:
                for backbone, info in BACKBONES.items():
                    out_dir = f"papers/modules/{module}/runs/rank_ablation/rank_{rank}/seed_{seed}/{backbone}"
                    run_one(module, backbone, info["family"], info["batch"], rank, 32, 100, seed, out_dir)
                    
        # Task A2: LoRA Rank Ablation
        for lora in LORA_RANKS:
            for backbone, info in BACKBONES.items():
                out_dir = f"papers/modules/{module}/runs/lora_ablation/lora_{lora}/{backbone}"
                run_one(module, backbone, info["family"], info["batch"], 64, lora, 100, 42, out_dir)
                
        # Task A3: Recovery Steps Ablation
        for steps in RECOVERY_STEPS:
            for backbone, info in BACKBONES.items():
                out_dir = f"papers/modules/{module}/runs/rec_ablation/steps_{steps}/{backbone}"
                run_one(module, backbone, info["family"], info["batch"], 64, 16, steps, 42, out_dir)

if __name__ == "__main__":
    main()
