import subprocess
import os
import sys
import time
import json
import shutil
import pandas as pd
from pathlib import Path

# Ensure online access for automated Hugging Face uploads
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

# Switch to project directory if on Windows GPU server
if os.path.exists(r"D:\svd_project"):
    os.chdir(r"D:\svd_project")

models = [
    {
        "name": "distilbert",
        "slug": "distilbert",
        "out_dir": r"results\unified_admm\distilbert",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\distilbert_sst2",
            "--model_family", "bert",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_admm\distilbert",
            "--parent_checkpoint", r"results\unified_asvd\distilbert\finetuned_model.pt",
            "--modules", "full,admm",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--admm_sparsity", "0.5",
            "--admm_quant_bits", "4",
            "--admm_structured_24", "true",
            "--admm_max_error", "0.45",
            "--admm_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_admm\distilbert\summary.csv",
        "stats_json": r"results\unified_admm\distilbert\admm_stats.json",
    },
    {
        "name": "roberta",
        "slug": "roberta",
        "out_dir": r"results\unified_admm\roberta",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\roberta_sst2",
            "--model_family", "roberta",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_admm\roberta",
            "--parent_checkpoint", r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,admm",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--admm_sparsity", "0.5",
            "--admm_quant_bits", "4",
            "--admm_structured_24", "true",
            "--admm_max_error", "0.45",
            "--admm_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_admm\roberta\summary.csv",
        "stats_json": r"results\unified_admm\roberta\admm_stats.json",
    },
    {
        "name": "gpt2",
        "slug": "gpt2",
        "out_dir": r"results\unified_admm\gpt2",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\gpt2_sst2",
            "--model_family", "gpt",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_admm\gpt2",
            "--parent_checkpoint", r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,admm",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--admm_sparsity", "0.5",
            "--admm_quant_bits", "4",
            "--admm_structured_24", "true",
            "--admm_max_error", "0.45",
            "--admm_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_admm\gpt2\summary.csv",
        "stats_json": r"results\unified_admm\gpt2\admm_stats.json",
    },
]

import argparse

parser = argparse.ArgumentParser(description="Unified 4-Model Hardened ADMM Benchmark")
parser.add_argument("--models", nargs="*", default=None, help="Specific models to run (e.g. gpt2 or distilbert,roberta)")
parser.add_argument("--modules", type=str, default=None, help="Comma-separated module names (e.g. full,admm)")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
parser.add_argument("--recovery_steps", type=int, default=None, help="Number of recovery fine-tuning steps")
parser.add_argument("--max_error", type=float, default=None, help="ADMM max reconstruction error threshold")
parser.add_argument("--sparsity", type=float, default=None, help="Target pruning sparsity")
parser.add_argument("--quant_bits", type=int, default=None, help="Quantization bitwidth")
parser.add_argument("--structured_24", type=str, default=None, help="Enable structured 2:4 sparsity (true/false)")
parser.add_argument("--attention_only", type=str, default=None, help="Target attention layers only (true/false)")
parser.add_argument("positional_models", nargs="*", default=None, help="Positional model names for backwards compatibility")

args = parser.parse_args()

all_benchmark_models = list(models)

selected_models = []
if args.models:
    for item in args.models:
        for sub in item.split(","):
            if sub.strip():
                selected_models.append(sub.strip())
elif args.positional_models:
    for item in args.positional_models:
        for sub in item.split(","):
            if sub.strip():
                selected_models.append(sub.strip())

if selected_models:
    selected_set = set(selected_models)
    models_to_run = [m for m in all_benchmark_models if m["name"] in selected_set]
else:
    models_to_run = list(all_benchmark_models)

def update_cmd_arg(cmd: list[str], flag: str, new_val: str):
    try:
        idx = cmd.index(flag)
        cmd[idx + 1] = str(new_val)
    except ValueError:
        cmd.extend([flag, str(new_val)])

for m in all_benchmark_models:
    if args.modules is not None:
        update_cmd_arg(m["cmd"], "--modules", args.modules)
    if args.batch_size is not None:
        update_cmd_arg(m["cmd"], "--batch_size", str(args.batch_size))
    if args.recovery_steps is not None:
        update_cmd_arg(m["cmd"], "--recovery_steps", str(args.recovery_steps))
    if args.max_error is not None:
        update_cmd_arg(m["cmd"], "--admm_max_error", str(args.max_error))
    if args.sparsity is not None:
        update_cmd_arg(m["cmd"], "--admm_sparsity", str(args.sparsity))
    if args.quant_bits is not None:
        update_cmd_arg(m["cmd"], "--admm_quant_bits", str(args.quant_bits))
    if args.structured_24 is not None:
        val = str(args.structured_24).strip().lower()
        if val in ("1", "true", "yes"):
            update_cmd_arg(m["cmd"], "--admm_structured_24", "true")
        elif val in ("0", "false", "no"):
            update_cmd_arg(m["cmd"], "--admm_structured_24", "false")
    if args.attention_only is not None:
        val = str(args.attention_only).strip().lower()
        if val in ("1", "true", "yes"):
            update_cmd_arg(m["cmd"], "--admm_target_attention_only", "true")
        elif val in ("0", "false", "no"):
            update_cmd_arg(m["cmd"], "--admm_target_attention_only", "false")

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL HARDENED ADMM BENCHMARK ON GPU SERVER", flush=True)
print(f"Executing models: {[m['name'] for m in models_to_run]}", flush=True)
print(f"Full benchmark suite: {[m['name'] for m in all_benchmark_models]}", flush=True)
print("=" * 80, flush=True)

# ------------------------------------------------------------------------------
# STAGE 1: EXECUTE ALL SPECIFIED BENCHMARK RUNS
# ------------------------------------------------------------------------------
for m in models_to_run:
    print(f"\n{'='*30} RUNNING MODEL: {m['name']} {'='*30}", flush=True)
    t0 = time.time()
    res = subprocess.run(m["cmd"])
    elapsed = time.time() - t0
    print(f"\n>>> FINISHED MODEL: {m['name']} in {elapsed:.1f}s (Exit Code: {res.returncode})", flush=True)

# ------------------------------------------------------------------------------
# COMPILE RESULTS SUMMARY FOR ALL 4 MODELS
# ------------------------------------------------------------------------------
final_stats = []

for m in all_benchmark_models:
    acc = None
    baseline_acc = None
    status = "UNKNOWN"
    num_compressed_layers = "N/A"
    f1 = None
    actual_sparsity = None
    quant_bits = 4
    recon_error = None
    peak_vram = None

    csv_path = Path(m["summary_csv"])
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                col = "accuracy_pct" if "accuracy_pct" in df.columns else ("mean_accuracy" if "mean_accuracy" in df.columns else None)
                if col:
                    row_base = df[df["module_name"] == "full"]
                    if not row_base.empty:
                        baseline_acc = float(row_base[col].values[-1])
                    row = df[df["module_name"] == "admm"]
                    if not row.empty:
                        acc = float(row[col].values[-1])
                        status = str(row["status"].values[-1])
                        if "f1_score" in row.columns and not pd.isna(row["f1_score"].values[-1]):
                            try: f1 = float(row["f1_score"].values[-1])
                            except Exception: pass
                        if "peak_vram_mb" in row.columns and not pd.isna(row["peak_vram_mb"].values[-1]):
                            try: peak_vram = float(row["peak_vram_mb"].values[-1])
                            except Exception: pass
                        if "num_compressed_layers" in row.columns and not pd.isna(row["num_compressed_layers"].values[-1]):
                            num_compressed_layers = str(row["num_compressed_layers"].values[-1])
                        if "avg_reconstruction_error" in row.columns and not pd.isna(row["avg_reconstruction_error"].values[-1]):
                            try: recon_error = float(row["avg_reconstruction_error"].values[-1])
                            except Exception: pass
        except Exception as e:
            print(f"Error reading summary CSV for {m['name']}: {e}", flush=True)

    stats_path = Path(m.get("stats_json", ""))
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as sf:
                s_data = json.load(sf)
                if num_compressed_layers in ("N/A", "0"):
                    num_compressed_layers = str(s_data.get("num_compressed", s_data.get("num_compressed_layers", num_compressed_layers)))
                if recon_error is None and "avg_reconstruction_error" in s_data:
                    recon_error = float(s_data["avg_reconstruction_error"])
                if "actual_sparsity" in s_data:
                    actual_sparsity = float(s_data["actual_sparsity"])
                if "quant_bits" in s_data:
                    quant_bits = int(s_data["quant_bits"])
        except Exception as e:
            print(f"Error reading stats JSON for {m['name']}: {e}", flush=True)

    delta_acc_str = f"{(acc - baseline_acc):+.2f}%" if (acc is not None and baseline_acc is not None) else "N/A"
    f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
    sparsity_str = f"{actual_sparsity*100:.1f}%" if actual_sparsity is not None else "50.0%"
    recon_str = f"{recon_error:.4f}" if recon_error is not None else "N/A"
    vram_str = f"{peak_vram:.1f} MB" if peak_vram is not None else "N/A"

    final_stats.append({
        "Model": m["name"],
        "Baseline Acc": f"{baseline_acc:.2f}%" if baseline_acc is not None else "N/A",
        "ADMM Acc": f"{acc:.2f}%" if acc is not None else "N/A",
        "Delta Acc": delta_acc_str,
        "F1": f1_str,
        "Status": status,
        "Compressed Layers": num_compressed_layers,
        "Actual Sparsity": sparsity_str,
        "Quant Bits": quant_bits,
        "Recon Error": recon_str,
        "Peak VRAM": vram_str,
    })

print("\n" + "=" * 80, flush=True)
print("4-MODEL BENCHMARK RESULTS SUMMARY (ADMM on GPU)", flush=True)
print("=" * 80, flush=True)
res_df = pd.DataFrame(final_stats)
print(res_df.to_string(index=False), flush=True)
print("=" * 80, flush=True)

# Save consolidated summary
out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "unified_admm_4models_summary.csv"
out_json = out_dir / "unified_admm_4models_summary.json"
res_df.to_csv(out_csv, index=False)
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(final_stats, f, indent=2)
print(f"\nSaved consolidated summaries to:\n- {out_csv}\n- {out_json}", flush=True)

# Generate and save comparative plots in results directory
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    _mod_slug = res_df.columns[2].replace(" Acc", "").lower() if len(res_df.columns) > 2 else "module"
    _plots_dir = out_dir / f"unified_{_mod_slug}"
    _plots_dir.mkdir(parents=True, exist_ok=True)

    _p_models = [str(s.get("Model", "")) for s in final_stats]
    _b_accs = [float(str(s.get("Baseline Acc", "0")).replace("%", "")) if str(s.get("Baseline Acc", "N/A")) != "N/A" else 0.0 for s in final_stats]
    _m_acc_col = [k for k in final_stats[0].keys() if "Acc" in k and k not in ("Baseline Acc", "Delta Acc")][0] if final_stats else "Acc"
    _m_accs = [float(str(s.get(_m_acc_col, "0")).replace("%", "")) if str(s.get(_m_acc_col, "N/A")) != "N/A" else 0.0 for s in final_stats]
    _c_rates = [float(str(s.get("Comp Rate", "1.0")).replace("x", "")) if str(s.get("Comp Rate", "N/A")) != "N/A" else 1.0 for s in final_stats]

    # Chart 1: Accuracy Comparison
    _x = np.arange(len(_p_models))
    _w = 0.35
    _fig, _ax1 = plt.subplots(figsize=(9, 5))
    _r1 = _ax1.bar(_x - _w/2, _b_accs, _w, label='Baseline Acc', color='#4C72B0')
    _r2 = _ax1.bar(_x + _w/2, _m_accs, _w, label=f'{_mod_slug.upper()} Acc', color='#55A868')
    _ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    _ax1.set_title(f'Unified 4-Model Quality Evaluation ({_mod_slug.upper()})', fontweight='bold')
    _ax1.set_xticks(_x)
    _ax1.set_xticklabels(_p_models, fontweight='bold')
    _ax1.set_ylim(0, 105)
    _ax1.legend(loc='upper right')
    _ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for _rect in _r1:
        _h = _rect.get_height()
        if _h > 0: _ax1.annotate(f'{_h:.1f}%', xy=(_rect.get_x() + _rect.get_width()/2, _h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for _rect in _r2:
        _h = _rect.get_height()
        if _h > 0: _ax1.annotate(f'{_h:.1f}%', xy=(_rect.get_x() + _rect.get_width()/2, _h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout()
    plt.savefig(_plots_dir / f"{_mod_slug}_4models_accuracy_comparison.png", dpi=300)
    plt.close()

    # Chart 2: Pareto Frontier
    _fig, _ax2 = plt.subplots(figsize=(8, 5))
    for _i, _m_name in enumerate(_p_models):
        if _m_accs[_i] > 0:
            _ax2.scatter(_c_rates[_i], _m_accs[_i], s=160, label=_m_name, zorder=5)
            _ax2.annotate(f"{_m_name} ({_c_rates[_i]:.2f}x, {_m_accs[_i]:.1f}%)", (_c_rates[_i], _m_accs[_i]), xytext=(7, -4), textcoords='offset points', fontsize=9)
    _ax2.set_xlabel('Compression Rate (x)', fontweight='bold')
    _ax2.set_ylabel('Accuracy (%)', fontweight='bold')
    _ax2.set_title(f'Pareto Frontier: Accuracy vs Compression ({_mod_slug.upper()})', fontweight='bold')
    _ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(_plots_dir / f"{_mod_slug}_pareto_accuracy_vs_compression.png", dpi=300)
    plt.close()
    print(f"Generated and saved comparative visualization charts to {_plots_dir}", flush=True)
except Exception as _e_p:
    print(f"[WARN] Could not generate comparative plots: {_e_p}", flush=True)


# ------------------------------------------------------------------------------
# STAGE 2: BATCH QUALITY GATE & DEFERRED EXPORT TO HUGGING FACE & GITHUB
# ------------------------------------------------------------------------------
print("\n" + "=" * 80, flush=True)
print("STAGE 2: QUALITY-GATED EXPORT TO HUGGING FACE & GITHUB (ADMM)", flush=True)
print("=" * 80, flush=True)

from Codes.framework.utils.hf_git_exporter import (
    find_documentation_file,
    upload_to_huggingface,
    update_documentation_table,
    commit_and_push_git,
)

hf_org = "Hooshaai"
doc_path = find_documentation_file("admm")
pass_count = 0
uploaded_repos = []

for m in all_benchmark_models:
    name = m["name"]
    slug = m.get("slug", name)
    out_m_dir = Path(m["out_dir"])
    weights_dir = out_m_dir / "checkpoints" / "admm"
    hf_export_dir = weights_dir / "hf_export"

    csv_path = Path(m["summary_csv"])
    row = {}
    baseline_acc_val = None
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            row_base = df[df["module_name"] == "full"]
            if not row_base.empty:
                baseline_acc_val = float(row_base["accuracy_pct"].values[-1] if "accuracy_pct" in row_base else row_base["mean_accuracy"].values[-1])
            row_match = df[df["module_name"] == "admm"]
            if not row_match.empty:
                row = row_match.iloc[-1].to_dict()
        except Exception as e:
            print(f"Error reading summary for {name}: {e}", flush=True)

    status = str(row.get("status", "FAIL"))
    try:
        acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0.0)))
    except Exception:
        acc = 0.0

    is_successful = (status == "PASS") and (acc >= 56.0)

    if not is_successful:
        print(f"[Quality Gate: REJECTED] {name} did not meet criteria (status={status}, acc={acc:.2f}% < 56.0%).", flush=True)
        if weights_dir.exists():
            print(f"  Pruning local checkpoint artifacts in {weights_dir}", flush=True)
            shutil.rmtree(weights_dir, ignore_errors=True)
        continue

    pass_count += 1
    print(f"[Quality Gate: PASSED] {name} passed criteria (status={status}, acc={acc:.2f}% >= 56.0%). Packaging for export...", flush=True)

    if not hf_export_dir.exists():
        print(f"  Warning: hf_export directory not found at {hf_export_dir}. Skipping HF upload.", flush=True)
        continue

    # Ensure lean payload (~90 MB)
    bin_file = hf_export_dir / "pytorch_model.bin"
    if bin_file.exists():
        try:
            bin_file.unlink()
            print(f"  Pruned uncompressed {bin_file.name} for lean ~90 MB payload", flush=True)
        except Exception:
            pass

    repo_id = f"{hf_org}/svd-linear-attention-{slug}-admm"
    perf_metrics = {
        "accuracy_pct": acc,
        "f1_score": row.get("f1_score", "N/A"),
        "precision": row.get("precision", "N/A"),
        "recall": row.get("recall", "N/A"),
        "val_loss": row.get("val_loss", "N/A"),
        "peak_vram_mb": row.get("peak_vram_mb", "N/A"),
        "tokens_per_sec": row.get("tokens_per_sec", "N/A"),
        "status": status,
    }
    comp_metrics = {
        "compression_rate": row.get("compression_rate", "1.0"),
        "compression_ratio_pct": row.get("compression_ratio_pct", "0.0"),
        "flops_reduction_pct": row.get("flops_reduction_pct", "0.0"),
        "num_compressed_layers": row.get("num_compressed_layers", "N/A"),
        "num_fallback_layers": row.get("num_fallback_layers", "0"),
        "avg_reconstruction_error": row.get("avg_reconstruction_error", "N/A"),
    }

    print(f"  Uploading lean package to Hugging Face: https://huggingface.co/{repo_id} (private)", flush=True)
    hf_url = upload_to_huggingface(
        export_dir=hf_export_dir,
        repo_id=repo_id,
        model_name=slug,
        module_name="admm",
        dataset_name="sst2",
        perf_metrics=perf_metrics,
        comp_metrics=comp_metrics,
    )
    if hf_url:
        print(f"  Successfully uploaded to {hf_url}", flush=True)
        uploaded_repos.append(hf_url)
    else:
        print(f"  Hugging Face upload failed or skipped for {repo_id}", flush=True)

    # Update documentation table
    if doc_path and doc_path.exists():
        f1_val = row.get("f1_score", "N/A")
        comp_val = row.get("compression_rate", "1.00")
        compressed_layers_val = row.get("num_compressed_layers", "N/A")
        update_documentation_table(
            doc_path=doc_path,
            model_name=slug,
            module_name="admm",
            dataset_name="sst2",
            status=status,
            baseline_acc=baseline_acc_val,
            acc=acc,
            f1=f1_val,
            comp_rate=comp_val,
            num_compressed=compressed_layers_val,
            hf_repo_url=hf_url or f"https://huggingface.co/{repo_id}",
        )
        print(f"  Updated documentation table in {doc_path.name}", flush=True)

# Step 4: Clean Git Commit and Push to Dedicated Branch
target_branch = "benchmark-results-admm"
print(f"\nCommitting results and documentation updates to dedicated branch '{target_branch}'...", flush=True)
commit_msg = f"feat(benchmark): ADMM 4-model evaluation ({pass_count}/{len(all_benchmark_models)} passed on SST-2)"
# push_res = commit_and_push_git(commit_msg, branch=target_branch)

if push_res:
    print(f"Git commit and push completed successfully on branch '{target_branch}': {push_res}", flush=True)
else:
    print(f"Git commit or push had issues. Working tree status preserved.", flush=True)

print("\n" + "=" * 80, flush=True)
print("ALL TASKS COMPLETED!", flush=True)
print(f"Passed Models: {pass_count} / {len(all_benchmark_models)}")
print(f"Uploaded Repositories: {len(uploaded_repos)}")
for u in uploaded_repos:
    print(f"  - {u}")
print("=" * 80, flush=True)
