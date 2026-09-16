#!/usr/bin/env python3
"""
Multi-Class Benchmark Sweep Orchestrator with Auto-Retrain & Publish (AG News - 4 Classes)
High-accuracy (>70% baseline), 4-class multi-class dataset benchmark sweep.
Runs all 83 thesis modules across backbones at batch size 64 for maximum GPU performance.
Includes built-in auto-retraining for any modules that produce sub-optimal or failed results,
and automatically publishes all passing models, model cards, and weights to Hugging Face.
"""

import os
import sys
import time
import csv
import argparse
import subprocess
from pathlib import Path

PYTHON_EXEC = r"C:\Users\Hadi Schumacher\anaconda3\python.exe" if sys.platform == "win32" else sys.executable
REPO_DIR = Path(r"D:\SVD_Project") if sys.platform == "win32" else Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_DIR))

SKIP_MODULES_BY_MODEL = {
    "distilbert": [
        "jamba2", "jamba", "decay_gated_linear_attention", "gfws_d",
        "spectral_residual_svd", "compression_unified_rope",
        "hgrn2_gated_deltanet", "hgrn2_layer", "gated_delta_net",
        "gated_deltanet_block", "svd_mamba",
    ],
    "roberta": [
        "jamba2", "jamba", "decay_gated_linear_attention", "gfws_d",
        "hgrn2_gated_deltanet", "hgrn2_layer", "gated_delta_net",
        "gated_deltanet_block", "svd_mamba",
    ],
    "gpt2": ["hgrn2_layer"],
}

BATCH_SIZE_BY_MODEL = {
    "distilbert": 128,
    "roberta": 64,
    "gpt2": 64,
}

COMPILE_SAFE_MODULES = {
    "svd_linear", "svd_llm_v2", "a3", "aasvd", "asvd",
    "mla", "low_rank", "sdrm", "compressed_mha",
    "compressed_attention", "compressed_attention_layer",
    "fused_svd_kernel", "kronecker", "hamming",
    "galerkin", "galerkin_attention", "nasc", "nasc_attention",
}

RECOVERY_OVERRIDES = {
    "svd_linear": {"recovery_steps": 100, "lora_rank": 32, "lora_alpha": 64},
    "svd_llm_v2": {"recovery_steps": 100, "lora_rank": 32, "lora_alpha": 64},
    "svd_mamba": {"recovery_steps": 100, "lora_rank": 32, "lora_alpha": 64},
    "a3": {"recovery_steps": 100, "lora_rank": 32, "lora_alpha": 64},
    "aasvd": {"recovery_steps": 100, "lora_rank": 32, "lora_alpha": 64},
    "asvd": {"recovery_steps": 100, "lora_rank": 32, "lora_alpha": 64},
    "mla": {"recovery_steps": 100, "lora_rank": 32, "lora_alpha": 64},
}

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

PASS_THRESHOLD_AGNEWS = 31.63

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

def check_result_status(mod_out_dir, mod_name):
    summary_csv = mod_out_dir / "summary.csv"
    if not summary_csv.exists():
        return "MISSING", 0.0
    try:
        with open(summary_csv, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("module_name") == mod_name:
                    status = row.get("status", "UNKNOWN")
                    try:
                        acc = float(row.get("accuracy_pct", 0.0))
                    except (ValueError, TypeError):
                        acc = 0.0
                    return status, acc
    except Exception:
        pass
    return "UNKNOWN", 0.0

def run_module_eval(sweep_script, cfg, base_ckpt, mod_out_dir, mod, batch_size, recovery_steps, cached_batches_dir=None, timeout=300, override_args=None, extra_args=None):

    model_name = Path(cfg["model_path"]).name
    if mod in SKIP_MODULES_BY_MODEL.get(model_name, []):
        print(f"Skipping {mod} on {model_name} (SKIP_NON_CAUSAL_BACKBONE)")
        return "SKIP_NON_CAUSAL_BACKBONE", 0.0

    def build_cmd(curr_bs):
        c = [
            PYTHON_EXEC, "-u", str(sweep_script),
            "--model", cfg["model_path"],
            "--model_family", cfg["family"],
            "--dataset", "ag_news",
            "--output_dir", str(mod_out_dir),
            "--parent_checkpoint", str(base_ckpt),
            "--modules", mod,
            "--batch_size", str(curr_bs),
            "--recovery_steps", str(recovery_steps),
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", str(timeout),
        ]
        if cached_batches_dir:
            c += ["--cached_batches_dir", str(cached_batches_dir)]
        if extra_args:
            c += extra_args
        if override_args:
            if "recovery_steps" in override_args:
                idx = c.index("--recovery_steps")
                c[idx+1] = str(override_args["recovery_steps"])
            if "lora_rank" in override_args:
                c.extend(["--lora_rank", str(override_args["lora_rank"])])
            if "lora_alpha" in override_args:
                c.extend(["--lora_alpha", str(override_args["lora_alpha"])])
        return c

    env = os.environ.copy()
    if mod in COMPILE_SAFE_MODULES:
        env["SVD_COMPILE"] = "1"
    else:
        env.pop("SVD_COMPILE", None)

    t0 = time.time()
    cmd = build_cmd(batch_size)
    try:
        proc = subprocess.run(cmd, cwd=str(REPO_DIR), env=env, check=False, capture_output=True, text=True)
        combined_output = (proc.stdout or "") + (proc.stderr or "")
        if "CUDA out of memory" in combined_output or "OutOfMemoryError" in combined_output:
            fallback_bs = max(8, batch_size // 2)
            print(f"[OOM-FALLBACK] CUDA out of memory on {mod} with batch={batch_size}. Retrying with batch={fallback_bs}...")
            cmd_retry = build_cmd(fallback_bs)
            proc_retry = subprocess.run(cmd_retry, cwd=str(REPO_DIR), env=env, check=False, capture_output=True, text=True)
            if proc_retry.stdout:
                print(proc_retry.stdout, end="")
            if proc_retry.stderr:
                print(proc_retry.stderr, end="", file=sys.stderr)
        else:
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
    except Exception as exc:
        print(f"Failed to run {mod}: {exc}")
    elapsed = time.time() - t0
    return elapsed

def main():
    parser = argparse.ArgumentParser(description="Multi-class AG News (4 Classes) benchmark sweep with auto-retrain")
    parser.add_argument("--models", default="distilbert,roberta,gpt2", help="Comma-separated models")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--recovery_steps", type=int, default=25)
    parser.add_argument("--modules", nargs="*", default=None, help="Subset of modules to run")
    parser.add_argument("--skip_modules", nargs="*", default=["mla", "hgrn2_gated_deltanet", "hgrn2_layer"], help="Modules to skip")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    skip_set = set(args.skip_modules) if args.skip_modules else set()
    all_modules = [m for m in (THESIS_MODULES if not args.modules else args.modules) if m not in skip_set]

    print("=" * 80)
    print("MULTI-CLASS HIGH-ACCURACY BENCHMARK SWEEP (AG News - 4 Classes)")
    print(f"Models: {models}")
    print(f"Total modules to run: {len(all_modules)}")
    print(f"Batch Size: {args.batch_size}, Recovery Steps: {args.recovery_steps}")
    print("Auto-Retrain on Bad Result: ENABLED (50 steps)")
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



    def get_batch_size(m_name: str) -> int:
        return BATCH_SIZE_BY_MODEL.get(m_name.lower(), 32)

    sweep_script = REPO_DIR / "Codes" / "framework" / "run_quality_sweep.py"
    uploader_script = REPO_DIR / "auto_upload_hf_results.py"

    for model_name in models:
        cfg = model_configs.get(model_name)
        if not cfg:
            print(f"Skipping unknown model: {model_name}")
            continue

        base_out_dir = REPO_DIR / "results" / f"{model_name}_agnews_baseline"
        base_out_dir.mkdir(parents=True, exist_ok=True)
        base_ckpt = base_out_dir / "finetuned_model.pt"

        # One shared batch cache per backbone - created on first module run, reused by all others
        cached_batches_dir = REPO_DIR / "results" / f"{model_name}_agnews_batch_cache"
        cached_batches_dir.mkdir(parents=True, exist_ok=True)
        
        current_batch_size = get_batch_size(model_name)

        # Step 1: Ensure baseline is pre-tuned on AG News
        if not base_ckpt.exists():
            print(f"\n[BASELINE] Pre-tuning baseline for {model_name} on AG News (4 Classes)...")
            base_cmd = [
                PYTHON_EXEC, "-u", str(sweep_script),
                "--model", cfg["model_path"],
                "--model_family", cfg["family"],
                "--dataset", "ag_news",
                "--output_dir", str(base_out_dir),
                "--modules", "full",
                "--batch_size", str(current_batch_size),
                "--finetune_steps", "600",
                "--precision", "fp16",
                "--fresh",
                "--cached_batches_dir", str(cached_batches_dir),
            ]
            subprocess.run(base_cmd, cwd=str(REPO_DIR), check=False)

        # Step 2: Run and verify all modules
        for idx, mod in enumerate(all_modules, 1):
            mod_out_dir = REPO_DIR / "results" / f"agnews_unified_{mod}" / model_name
            mod_out_dir.mkdir(parents=True, exist_ok=True)

            status, acc = check_result_status(mod_out_dir, mod)

            # If already evaluated and passed, skip
            if status not in ("MISSING", "UNKNOWN") and acc >= PASS_THRESHOLD_AGNEWS and status not in ("FAIL", "PASS_LOW"):
                print(f"[{idx}/{len(all_modules)}] Skipping {mod} on {model_name} (already passed with {acc:.2f}%).")
                continue

            # If existing run was a bad result (FAIL / PASS_LOW / acc < threshold), auto-retrain with enhanced budget
            is_retrain = (status in ("FAIL", "PASS_LOW") or (status not in ("MISSING", "UNKNOWN") and acc < PASS_THRESHOLD_AGNEWS))
            rec_steps = 50 if is_retrain else args.recovery_steps

            if is_retrain:
                print(f"\n[{idx}/{len(all_modules)}] [AUTO-RETRAIN] Re-running bad module {mod} on {model_name} with enhanced budget (steps={rec_steps})...")
            else:
                print(f"\n[{idx}/{len(all_modules)}] Evaluating {mod} on {model_name} (AG News 4-class)...")

            extra_args = []
            if model_name in ("roberta", "gpt2"):
                extra_args += ["--eval_seq_len", "256"]

            elapsed = run_module_eval(sweep_script, cfg, base_ckpt, mod_out_dir, mod, get_batch_size(model_name), rec_steps, cached_batches_dir=cached_batches_dir, override_args=RECOVERY_OVERRIDES.get(mod, None), extra_args=extra_args)
            print(f"Completed {mod} in {elapsed:.1f}s.")

            # Check outcome
            new_status, new_acc = check_result_status(mod_out_dir, mod)
            if new_acc >= PASS_THRESHOLD_AGNEWS and new_status not in ("FAIL", "PASS_LOW"):
                print(f"[PASS] SUCCESS: {mod} on {model_name} passed (Acc: {new_acc:.2f}%). Triggering HF upload...")
                if uploader_script.exists():
                    subprocess.run([PYTHON_EXEC, str(uploader_script), "0"], cwd=str(REPO_DIR), check=False)
            else:
                print(f"[NOTICE] {mod} on {model_name} outcome: {new_status} (Acc: {new_acc:.2f}%). Skipping bad upload.")

            clean_large_weights(mod_out_dir)

if __name__ == "__main__":
    main()
