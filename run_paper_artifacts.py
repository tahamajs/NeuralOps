import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import time

# Configured for Task 4: Capped Ablation Scope
MAX_MODULES = 20
MAX_RANKS_PER_MODULE = 4          # [32, 64, 128, 256]
SEEDS_PER_RUN = 1                 # just seed 42 for now
BACKBONES = ["distilbert"]        # expand only after pilot succeeds

RANKS = [32, 64, 128, 256]
SEEDS = [42]
BACKBONE_CONFIGS = {
    "distilbert": {
        "family": "bert",
        "batch": 32,
        "model_path": "models/distilbert_sst2",
        "parent_checkpoint": "results/distilbert_agnews_baseline/finetuned_model.pt",
    },
    "roberta": {
        "family": "roberta",
        "batch": 16,
        "model_path": "models/roberta-base",
        "parent_checkpoint": "results/roberta_agnews_baseline/finetuned_model.pt",
    },
    "gpt2": {
        "family": "gpt",
        "batch": 16,
        "model_path": "models/gpt2",
        "parent_checkpoint": "results/gpt2_agnews_baseline/finetuned_model.pt",
    },
}

RANK_ARG = {
    "svd_linear": "rank",
    "svd_llm_v2": "rank",
    "asvd": "asvd_rank",
    "aasvd": "aasvd_rank",
    "a3": "a3_rank",
    "low_rank": "rank",
}

def load_modules():
    mod_path = Path("papers/_rank_ablation/modules.txt")
    if mod_path.exists():
        with open(mod_path, "r", encoding="utf-8") as f:
            mods = [line.strip() for line in f if line.strip()]
        if mods:
            # Ensure svd_linear is pilot if present
            if "svd_linear" in mods:
                mods.remove("svd_linear")
                mods.insert(0, "svd_linear")
            return mods[:MAX_MODULES]
    return ["svd_linear"]

def run_cmd(cmd):
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=False)

def run_one(module, backbone, rank, lora_rank, rec_steps, seed, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if (Path(out_dir) / "summary.csv").exists():
        print(f"[CACHE] Skipping {out_dir}", flush=True)
        return
        
    info = BACKBONE_CONFIGS[backbone]
    ckpt = info["parent_checkpoint"]
    if not Path(ckpt).exists():
        fallback_ckpt = f"results/unified_asvd/{backbone}/finetuned_model.pt"
        if Path(fallback_ckpt).exists():
            ckpt = fallback_ckpt
        else:
            ckpt = None

    rank_flag = f"--{RANK_ARG.get(module, 'rank')}"
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
        rank_flag, str(rank),
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
    modules = load_modules()
    print("=" * 70, flush=True)
    print(f"Starting Capped Paper Artifacts Suite (Total modules: {len(modules)})", flush=True)
    print(f"Backbones: {BACKBONES} | Ranks: {RANKS} | Seeds: {SEEDS}", flush=True)
    print(f"Estimated total runs: {len(modules) * len(RANKS) * len(SEEDS) * len(BACKBONES)}", flush=True)
    print("=" * 70, flush=True)

    for idx, module in enumerate(modules, start=1):
        print(f"\n[{idx}/{len(modules)}] === Module: {module} ===", flush=True)
        # Task A1: Rank Ablation
        for rank in RANKS:
            for seed in SEEDS:
                for backbone in BACKBONES:
                    out_dir = f"papers/modules/{module}/runs/rank_ablation/rank_{rank}/seed_{seed}/{backbone}"
                    run_one(module, backbone, rank, 32, 100, seed, out_dir)

if __name__ == "__main__":
    main()
