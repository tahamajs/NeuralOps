#!/usr/bin/env python3
"""
Multi-Class Benchmark Sweep Orchestrator (GLUE MNLI - 3 Classes: entailment, neutral, contradiction)
Runs all 83 thesis modules across the specified backbones.
Uses calibrated recovery steps (15) and batch size (16) for high speed and throughput.
"""

import os
import sys
import time
import argparse
import subprocess
from pathlib import Path

PYTHON_EXEC = r"C:\Users\Hadi Schumacher\anaconda3\python.exe" if sys.platform == "win32" else sys.executable
REPO_DIR = Path(r"D:\SVD_Project") if sys.platform == "win32" else Path(__file__).resolve().parents[0]

sys.path.insert(0, str(REPO_DIR))

# All 83 canonical thesis modules
THESIS_MODULES = [
    "fusion_attn", "primal", "gqa", "adaptive_sparse_attention", "adaptive_span",
    "adasplash", "goldfinch_ahn", "halo_regla", "nsa_attention", "nha",
    "squeezing_heads_distiller", "progressive_token_pruning", "eat", "infini_attention",
    "optimized_hybrid_attention", "asvd", "rpc", "svd_mamba", "svd_linear",
    "res_svd", "minimal_aurora", "minimal_nova", "minimal_prism", "minimal_svd",
    "svd_llm_v2", "mla", "dobi", "nystrom", "unified_rope", "hgrn2_gated_deltanet",
    "hgrn2_layer", "soafusedattention", "adaptive_svd_attention", "bam_umoe",
    "block_transformer_hiba", "chela_slay", "cluster", "cosformer",
    "decay_linear_attention", "expert_choice_moe", "flash3", "flash3_fp8",
    "galerkin_attention", "gated_delta_net", "gated_deltanet_block", "griffin_ssd",
    "hymba_transx", "jamba_info", "vspec", "leka_v1", "leka_v2", "leka_v3",
    "linformer", "tpa", "lora_attention", "xattention", "mamba", "mixattention_moska",
    "moe_deltakv_paged", "nasc_attention", "nexus", "orion", "padre", "performer",
    "phantom", "projection_factorized_attn", "rala", "reap_moe", "regla", "routing",
    "salt", "simple_ssm", "span_expanded", "svd_projection", "template_attention",
    "token_compressed", "moba", "spectra_v2", "compressed_mha", "svd_compressed",
    "spectron", "soa_module", "aegis_v5"
]

def clean_large_weights(out_dir):
    p = Path(out_dir)
    if not p.exists():
        return
    for f in p.rglob("*.pt"):
        if "finetuned_model.pt" not in f.name and "baseline" not in str(f).lower():
            try:
                f.unlink()
            except Exception:
                pass

def main():
    parser = argparse.ArgumentParser(description="Multi-class MNLI benchmark sweep for all 83 modules")
    parser.add_argument("--models", default="distilbert", help="Comma-separated models: distilbert, roberta, gpt2")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--recovery_steps", type=int, default=15)
    parser.add_argument("--modules", nargs="*", default=None, help="Subset of modules to run")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    all_modules = THESIS_MODULES if not args.modules else args.modules

    print("=" * 80)
    print(f"MULTI-CLASS (MNLI - 3 Classes) BENCHMARK SWEEP")
    print(f"Models: {models}")
    print(f"Total modules to run: {len(all_modules)}")
    print(f"Batch Size: {args.batch_size}, Recovery Steps: {args.recovery_steps}")
    print("=" * 80)

    model_configs = {
        "distilbert": {
            "model_path": "models/distilbert_sst2",
            "family": "bert",
        },
        "roberta": {
            "model_path": "models/roberta-base",
            "family": "roberta",
        },
        "gpt2": {
            "model_path": "models/gpt2",
            "family": "gpt",
        },
    }

    sweep_script = REPO_DIR / "Codes" / "framework" / "run_quality_sweep.py"

    for model_name in models:
        cfg = model_configs.get(model_name)
        if not cfg:
            print(f"Skipping unknown model: {model_name}")
            continue

        base_out_dir = REPO_DIR / "results" / f"{model_name}_mnli_baseline"
        base_out_dir.mkdir(parents=True, exist_ok=True)
        base_ckpt = base_out_dir / "finetuned_model.pt"

        # Step 1: Ensure baseline is trained/evaluated on MNLI
        if not base_ckpt.exists():
            print(f"\n[BASELINE] Pre-tuning baseline for {model_name} on MNLI...")
            base_cmd = [
                PYTHON_EXEC, "-u", str(sweep_script),
                "--model", cfg["model_path"],
                "--model_family", cfg["family"],
                "--dataset", "mnli",
                "--output_dir", str(base_out_dir),
                "--modules", "full",
                "--batch_size", str(args.batch_size),
                "--finetune_steps", "200",
                "--precision", "fp16",
                "--fresh",
            ]
            subprocess.run(base_cmd, cwd=str(REPO_DIR), check=False)

        # Step 2: Run all modules
        for idx, mod in enumerate(all_modules, 1):
            mod_out_dir = REPO_DIR / "results" / f"mnli_unified_{mod}" / model_name
            mod_out_dir.mkdir(parents=True, exist_ok=True)

            summary_csv = mod_out_dir / "summary.csv"
            if summary_csv.exists():
                print(f"[{idx}/{len(all_modules)}] Skipping {mod} on {model_name} (already evaluated).")
                continue

            print(f"\n[{idx}/{len(all_modules)}] Evaluating {mod} on {model_name} (MNLI)...")
            cmd = [
                PYTHON_EXEC, "-u", str(sweep_script),
                "--model", cfg["model_path"],
                "--model_family", cfg["family"],
                "--dataset", "mnli",
                "--output_dir", str(mod_out_dir),
                "--parent_checkpoint", str(base_ckpt),
                "--modules", mod,
                "--batch_size", str(args.batch_size),
                "--recovery_steps", str(args.recovery_steps),
                "--precision", "fp16",
                "--fresh",
                "--defer_export",
                "--timeout", "300",
            ]
            t0 = time.time()
            try:
                subprocess.run(cmd, cwd=str(REPO_DIR), check=False)
            except Exception as exc:
                print(f"Failed to run {mod}: {exc}")
            elapsed = time.time() - t0
            print(f"Completed {mod} in {elapsed:.1f}s.")
            clean_large_weights(mod_out_dir)

if __name__ == "__main__":
    main()
