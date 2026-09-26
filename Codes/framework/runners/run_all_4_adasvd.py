import subprocess
import os
import sys
import time
import json
import shutil
import pandas as pd
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Ensure online access for automated Hugging Face uploads
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

# Switch to project directory if on Windows GPU server
if os.path.exists(r"PROJECT_ROOT"):
    HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent.parent
if os.path.exists(r"PROJECT_ROOT"):
    os.chdir(r"PROJECT_ROOT")
else:
    os.chdir(PROJECT_ROOT)
RESULTS_DIR = Path("results").resolve()

models = [
    {
        "name": "distilbert",
        "slug": "distilbert",
        "out_dir": r"results\unified_adasvd\distilbert",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\distilbert_sst2",
            "--model_family", "bert",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_adasvd\distilbert",
            "--parent_checkpoint", r"results\unified_asvd\distilbert\finetuned_model.pt",
            "--modules", "full,adasvd",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--rank", "432",
            "--adasvd_max_error", "0.70",
            "--adasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_adasvd\distilbert\summary.csv",
        "stats_json": r"results\unified_adasvd\distilbert\adasvd_stats.json",
    },
    {
        "name": "roberta",
        "slug": "roberta",
        "out_dir": r"results\unified_adasvd\roberta",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\roberta_sst2",
            "--model_family", "roberta",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_adasvd\roberta",
            "--parent_checkpoint", r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,adasvd",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--rank", "432",
            "--adasvd_max_error", "0.70",
            "--adasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_adasvd\roberta\summary.csv",
        "stats_json": r"results\unified_adasvd\roberta\adasvd_stats.json",
    },
    {
        "name": "gpt2",
        "slug": "gpt2",
        "out_dir": r"results\unified_adasvd\gpt2",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\gpt2_sst2",
            "--model_family", "gpt",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_adasvd\gpt2",
            "--parent_checkpoint", r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,adasvd",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--rank", "432",
            "--adasvd_max_error", "0.70",
            "--adasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_adasvd\gpt2\summary.csv",
        "stats_json": r"results\unified_adasvd\gpt2\adasvd_stats.json",
    },
]

import argparse

parser = argparse.ArgumentParser(description="Unified 4-Model Hardened AdaSVD Benchmark")
parser.add_argument("--models", nargs="*", default=None, help="Specific models to run (e.g. gpt2 or distilbert,roberta)")
parser.add_argument("--modules", type=str, default=None, help="Comma-separated module names (e.g. full,adasvd)")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
parser.add_argument("--recovery_steps", type=int, default=None, help="Number of recovery fine-tuning steps")
parser.add_argument("--rank", type=int, default=None, help="Compression rank override")
parser.add_argument("--max_error", type=float, default=None, help="AdaSVD max reconstruction error threshold")
parser.add_argument("--attention_only", type=str, default=None, help="Target attention layers only (true/false)")
parser.add_argument("positional_models", nargs="*", default=None, help="Positional model names for backwards compatibility")

args = parser.parse_args()

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
    models = [m for m in models if m["name"] in selected_set]

def update_cmd_arg(cmd: list[str], flag: str, new_val: str):
    try:
        idx = cmd.index(flag)
        cmd[idx + 1] = str(new_val)
    except ValueError:
        cmd.extend([flag, str(new_val)])

for m in models:
    if args.modules is not None:
        update_cmd_arg(m["cmd"], "--modules", args.modules)
    if args.batch_size is not None:
        update_cmd_arg(m["cmd"], "--batch_size", str(args.batch_size))
    if args.recovery_steps is not None:
        update_cmd_arg(m["cmd"], "--recovery_steps", str(args.recovery_steps))
    if args.rank is not None:
        update_cmd_arg(m["cmd"], "--rank", str(args.rank))
    if args.max_error is not None:
        update_cmd_arg(m["cmd"], "--adasvd_max_error", str(args.max_error))
    if args.attention_only is not None:
        val = str(args.attention_only).strip().lower()
        if val in ("1", "true", "yes"):
            update_cmd_arg(m["cmd"], "--adasvd_target_attention_only", "true")
        elif val in ("0", "false", "no"):
            update_cmd_arg(m["cmd"], "--adasvd_target_attention_only", "false")

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL HARDENED AdaSVD BENCHMARK ON GPU SERVER", flush=True)
print(f"Running models: {[m['name'] for m in models]}", flush=True)
print("=" * 80, flush=True)

final_stats = []

for m in models:
    print(f"\n{'='*30} RUNNING MODEL: {m['name']} {'='*30}", flush=True)
    t0 = time.time()
    
    # Run directly without shell=True to avoid path space issues with Python binary on Windows
    res = subprocess.run(m["cmd"])
    elapsed = time.time() - t0
    
    acc = None
    baseline_acc = None
    status = "UNKNOWN" if res.returncode == 0 else f"EXIT_{res.returncode}"
    num_compressed_layers = "N/A"
    
    f1 = None
    comp_rate = None
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
                    row = df[df["module_name"] == "adasvd"]
                    if not row.empty:
                        acc = float(row[col].values[-1])
                        status = str(row["status"].values[-1])
                        if "f1_score" in row.columns and not pd.isna(row["f1_score"].values[-1]):
                            try: f1 = float(row["f1_score"].values[-1])
                            except Exception: pass
                        if "compression_rate" in row.columns and not pd.isna(row["compression_rate"].values[-1]):
                            try: comp_rate = float(row["compression_rate"].values[-1])
                            except Exception: pass
                        if "peak_vram_mb" in row.columns and not pd.isna(row["peak_vram_mb"].values[-1]):
                            try: peak_vram = float(row["peak_vram_mb"].values[-1])
                            except Exception: pass
                        if "num_compressed_layers" in row.columns and not pd.isna(row["num_compressed_layers"].values[-1]):
                            num_compressed_layers = str(row["num_compressed_layers"].values[-1])
        except Exception as e:
            print(f"Error reading summary CSV for {m['name']}: {e}", flush=True)

    stats_path = Path(m.get("stats_json", ""))
    if stats_path.exists() and num_compressed_layers == "N/A":
        try:
            with open(stats_path, "r", encoding="utf-8") as sf:
                s_data = json.load(sf)
                num_compressed_layers = s_data.get("num_compressed_layers", "N/A")
        except Exception as e:
            print(f"Error reading stats JSON for {m['name']}: {e}", flush=True)

    delta_acc_str = f"{(acc - baseline_acc):+.2f}%" if (acc is not None and baseline_acc is not None) else "N/A"
    f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
    comp_rate_str = f"{comp_rate:.2f}x" if comp_rate is not None else "1.00x"
    vram_str = f"{peak_vram:.1f} MB" if peak_vram is not None else "N/A"

    final_stats.append({
        "Model": m["name"],
        "Exit Code": res.returncode,
        "Baseline Acc": f"{baseline_acc:.2f}%" if baseline_acc is not None else "N/A",
        "AdaSVD Acc": f"{acc:.2f}%" if acc is not None else "N/A",
        "Delta Acc": delta_acc_str,
        "F1": f1_str,
        "Status": status,
        "Compressed Layers": num_compressed_layers,
        "Comp Rate": comp_rate_str,
        "Peak VRAM": vram_str,
        "Time (s)": f"{elapsed:.1f}",
    })
    print(f"\n>>> FINISHED MODEL: {m['name']} in {elapsed:.1f}s (Exit: {res.returncode}, Status: {status}, Compressed Layers: {num_compressed_layers}, Comp Rate: {comp_rate_str})", flush=True)

print("\n" + "=" * 80, flush=True)
print("4-MODEL BENCHMARK RESULTS SUMMARY (AdaSVD on GPU)", flush=True)
print("=" * 80, flush=True)
res_df = pd.DataFrame(final_stats)
print(res_df.to_string(index=False), flush=True)
print("=" * 80, flush=True)

# Save consolidated summary
out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "unified_adasvd_4models_summary.csv"
out_json = out_dir / "unified_adasvd_4models_summary.json"
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


# ==============================================================================
# BATCH QUALITY GATE & DEFERRED EXPORT
# ==============================================================================
print("\n" + "=" * 80, flush=True)
print("STAGE 2: QUALITY-GATED EXPORT TO HUGGING FACE & GITHUB (AdaSVD)", flush=True)
print("=" * 80, flush=True)

try:
    from Codes.framework.utils.hf_git_exporter import (
        find_documentation_file,
        upload_to_huggingface,
        update_documentation_table,
        commit_and_push_git,
    )
except Exception as _e_imp:
    print(f"[NOTE] Automated HF/Git export module not loaded: {_e_imp}", flush=True)
    find_documentation_file = None
    upload_to_huggingface = None
    update_documentation_table = None
    commit_and_push_git = None

hf_org = "Hooshaai"
doc_path = find_documentation_file("adasvd") if find_documentation_file else None
pass_count = 0
uploaded_repos = []

for m in models:
    name = m["name"]
    slug = m.get("slug", name)
    out_dir = Path(m["out_dir"])
    weights_dir = out_dir / "checkpoints" / "adasvd"
    hf_export_dir = weights_dir / "hf_export"

    csv_path = Path(m["summary_csv"])
    row = {}
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            row_match = df[df["module_name"] == "adasvd"]
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

    repo_id = f"{hf_org}/svd-linear-attention-{slug}-adasvd"
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
        module_name="adasvd",
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
        svd_layers = row.get("num_compressed_layers", "N/A")
        update_documentation_table(
            doc_path=doc_path,
            model_name=slug,
            module_name="adasvd",
            dataset_name="sst2",
            status=status,
            baseline_acc=None,
            acc=acc,
            f1=f1_val,
            comp_rate=comp_val,
            num_compressed=svd_layers,
            hf_repo_url=hf_url or f"https://huggingface.co/{repo_id}",
        )
        print(f"  Updated documentation table in {doc_path.name}", flush=True)

# Step 4: Clean Git Commit and Push to Dedicated Branch
target_branch = "benchmark-results-adasvd"
print(f"\nCommitting results and documentation updates to dedicated branch '{target_branch}'...", flush=True)
commit_msg = f"feat(benchmark): AdaSVD 4-model evaluation ({pass_count}/{len(models)} passed on SST-2)"
push_res = None
# push_res = commit_and_push_git(commit_msg, branch=target_branch)

if push_res:
    print(f"Git commit and push completed successfully on branch '{target_branch}': {push_res}", flush=True)
else:
    print(f"Git commit or push had issues. Working tree status preserved.", flush=True)

print("\n" + "=" * 80, flush=True)
print("ALL TASKS COMPLETED!", flush=True)
print(f"Passed Models: {pass_count} / {len(models)}")
print(f"Uploaded Repositories: {len(uploaded_repos)}")
for u in uploaded_repos:
    print(f"  - {u}")
print("=" * 80, flush=True)
