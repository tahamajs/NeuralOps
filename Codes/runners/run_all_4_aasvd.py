import subprocess
import os
import sys
import time
import json
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
        "out_dir": r"results\unified_aasvd\distilbert",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\distilbert_sst2",
            "--model_family", "bert",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_aasvd\distilbert",
            "--parent_checkpoint", r"results\unified_asvd\distilbert\finetuned_model.pt",
            "--modules", "full,aasvd",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--rank", "288",
            "--aasvd_max_error", "0.50",
            "--aasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_aasvd\distilbert\summary.csv",
        "stats_json": r"results\unified_aasvd\distilbert\aasvd_stats.json",
    },
    {
        "name": "roberta",
        "slug": "roberta",
        "out_dir": r"results\unified_aasvd\roberta",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\roberta_sst2",
            "--model_family", "roberta",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_aasvd\roberta",
            "--parent_checkpoint", r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,aasvd",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--rank", "288",
            "--aasvd_max_error", "0.50",
            "--aasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_aasvd\roberta\summary.csv",
        "stats_json": r"results\unified_aasvd\roberta\aasvd_stats.json",
    },
    {
        "name": "gpt2",
        "slug": "gpt2",
        "out_dir": r"results\unified_aasvd\gpt2",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\gpt2_sst2",
            "--model_family", "gpt",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_aasvd\gpt2",
            "--parent_checkpoint", r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,aasvd",
            "--batch_size", "8",
            "--recovery_steps", "100",
            "--learning_rate", "2e-4",
            "--rank", "288",
            "--aasvd_max_error", "0.75",
            "--aasvd_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_aasvd\gpt2\summary.csv",
        "stats_json": r"results\unified_aasvd\gpt2\aasvd_stats.json",
    },
]

import argparse

parser = argparse.ArgumentParser(description="Unified 4-Model Hardened AASVD Benchmark")
parser.add_argument("--models", nargs="*", default=None, help="Specific models to run (e.g. gpt2 or distilbert,roberta)")
parser.add_argument("--modules", type=str, default=None, help="Comma-separated module names (e.g. full,aasvd)")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
parser.add_argument("--recovery_steps", type=int, default=None, help="Number of recovery fine-tuning steps")
parser.add_argument("--rank", type=int, default=None, help="Compression rank override")
parser.add_argument("--max_error", type=float, default=None, help="AASVD max reconstruction error threshold")
parser.add_argument("--attention_only", type=str, default=None, help="Target attention layers only (true/false)")
parser.add_argument("--ema_beta", type=float, default=None, help="EMA beta momentum for covariance accumulation")
parser.add_argument("--fallback_to_original", type=str, default=None, help="Fallback to original weights (true/false)")
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
        update_cmd_arg(m["cmd"], "--aasvd_max_error", str(args.max_error))
    if args.attention_only is not None:
        val = str(args.attention_only).strip().lower()
        if val in ("1", "true", "yes"):
            update_cmd_arg(m["cmd"], "--aasvd_target_attention_only", "true")
        elif val in ("0", "false", "no"):
            update_cmd_arg(m["cmd"], "--aasvd_target_attention_only", "false")
    if args.ema_beta is not None:
        update_cmd_arg(m["cmd"], "--aasvd_ema_beta", str(args.ema_beta))
    if args.fallback_to_original is not None:
        val = str(args.fallback_to_original).strip().lower()
        if val in ("1", "true", "yes"):
            update_cmd_arg(m["cmd"], "--aasvd_fallback_to_original", "true")
        elif val in ("0", "false", "no"):
            update_cmd_arg(m["cmd"], "--aasvd_fallback_to_original", "false")

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL HARDENED AASVD BENCHMARK ON GPU SERVER", flush=True)
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
                    row = df[df["module_name"] == "aasvd"]
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
                        num_fallback_layers = str(row.get("num_fallback_layers", pd.Series(["0"])).values[-1]) if "num_fallback_layers" in row.columns else "0"
                        avg_error = str(row.get("avg_reconstruction_error", pd.Series(["N/A"])).values[-1]) if "avg_reconstruction_error" in row.columns else "N/A"
                        val_loss = float(row["val_loss"].values[-1]) if "val_loss" in row.columns and not pd.isna(row["val_loss"].values[-1]) else None
                        precision = float(row["precision"].values[-1]) if "precision" in row.columns and not pd.isna(row["precision"].values[-1]) else None
                        recall = float(row["recall"].values[-1]) if "recall" in row.columns and not pd.isna(row["recall"].values[-1]) else None
                        tokens_per_sec = float(row["tokens_per_sec"].values[-1]) if "tokens_per_sec" in row.columns and not pd.isna(row["tokens_per_sec"].values[-1]) else None
                        flops_reduction = float(row["flops_reduction_pct"].values[-1]) if "flops_reduction_pct" in row.columns and not pd.isna(row["flops_reduction_pct"].values[-1]) else None
                        eval_time = float(row["eval_time_s"].values[-1]) if "eval_time_s" in row.columns and not pd.isna(row["eval_time_s"].values[-1]) else None
                        latency = float(row["latency_ms"].values[-1]) if "latency_ms" in row.columns and not pd.isna(row["latency_ms"].values[-1]) else None
                        if "elapsed_s" in row.columns and not pd.isna(row["elapsed_s"].values[-1]):
                            try:
                                elapsed = float(row["elapsed_s"].values[-1])
                            except Exception:
                                pass
        except Exception as e:
            print(f"Error reading summary CSV for {m['name']}: {e}", flush=True)

    stats_path = Path(m.get("stats_json", ""))
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as sf:
                s_data = json.load(sf)
                if num_compressed_layers == "N/A":
                    num_compressed_layers = s_data.get("num_compressed_layers", "N/A")
                if 'num_fallback_layers' not in locals() or num_fallback_layers == "0":
                    num_fallback_layers = str(s_data.get("num_fallback_layers", num_fallback_layers if 'num_fallback_layers' in locals() else "0"))
                if 'avg_error' not in locals() or avg_error == "N/A":
                    avg_error = str(s_data.get("avg_reconstruction_error", avg_error if 'avg_error' in locals() else "N/A"))
        except Exception as e:
            print(f"Error reading stats JSON for {m['name']}: {e}", flush=True)

    delta_acc_str = f"{(acc - baseline_acc):+.2f}%" if (acc is not None and baseline_acc is not None) else "N/A"
    f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
    comp_rate_str = f"{comp_rate:.2f}x" if comp_rate is not None else "1.00x"
    vram_str = f"{peak_vram:.1f} MB" if peak_vram is not None else "N/A"
    num_fallback_str = num_fallback_layers if 'num_fallback_layers' in locals() else "0"
    avg_error_str = avg_error if 'avg_error' in locals() else "N/A"
    inf_time_str = f"{eval_time:.2f}s" if ('eval_time' in locals() and eval_time is not None) else f"{elapsed:.1f}s"
    latency_str = f"{latency:.2f} ms" if ('latency' in locals() and latency is not None) else "N/A"

    final_stats.append({
        "Model": m["name"],
        "Exit Code": res.returncode,
        "Baseline Acc": f"{baseline_acc:.2f}%" if baseline_acc is not None else "N/A",
        "AASVD Acc": f"{acc:.2f}%" if acc is not None else "N/A",
        "Delta Acc": delta_acc_str,
        "F1": f1_str,
        "Precision": f"{precision:.4f}" if ('precision' in locals() and precision is not None) else "N/A",
        "Recall": f"{recall:.4f}" if ('recall' in locals() and recall is not None) else "N/A",
        "Val Loss": f"{val_loss:.4f}" if ('val_loss' in locals() and val_loss is not None) else "N/A",
        "Status": status,
        "Compressed Layers": num_compressed_layers,
        "Fallback Layers": num_fallback_str,
        "Avg Error": avg_error_str,
        "Comp Rate": comp_rate_str,
        "FLOPs Red": f"{flops_reduction:.1f}%" if ('flops_reduction' in locals() and flops_reduction is not None) else "N/A",
        "Peak VRAM": vram_str,
        "Inference Time": inf_time_str,
        "Latency / sample": latency_str,
        "Throughput (tok/s)": f"{tokens_per_sec:.1f}" if ('tokens_per_sec' in locals() and tokens_per_sec is not None) else "N/A",
    })
    print(f"\n>>> FINISHED MODEL: {m['name']} in Inference: {inf_time_str} (Exit: {res.returncode}, Status: {status}, Compressed Layers: {num_compressed_layers}, Comp Rate: {comp_rate_str})", flush=True)

print("\n" + "=" * 80, flush=True)
print("4-MODEL BENCHMARK RESULTS SUMMARY (AASVD on GPU)", flush=True)
print("=" * 80, flush=True)
res_df = pd.DataFrame(final_stats)
print(res_df.to_string(index=False), flush=True)
print("=" * 80, flush=True)

# Save consolidated summary
out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "unified_aasvd_4models_summary.csv"
out_json = out_dir / "unified_aasvd_4models_summary.json"
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

    plots_dir = out_dir / "unified_aasvd"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_models = [s["Model"] for s in final_stats]
    base_accs = [float(s["Baseline Acc"].replace("%", "")) if s["Baseline Acc"] != "N/A" else 0.0 for s in final_stats]
    aasvd_accs = [float(s["AASVD Acc"].replace("%", "")) if s["AASVD Acc"] != "N/A" else 0.0 for s in final_stats]
    f1_scores = [float(s["F1"]) if s["F1"] != "N/A" else 0.0 for s in final_stats]
    comp_rates = [float(s["Comp Rate"].replace("x", "")) if s["Comp Rate"] != "N/A" else 1.0 for s in final_stats]

    # Chart 1: Accuracy & F1 Comparison
    x = np.arange(len(plot_models))
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(10, 6))
    rects1 = ax1.bar(x - width/2, base_accs, width, label='Baseline Acc', color='#4C72B0')
    rects2 = ax1.bar(x + width/2, aasvd_accs, width, label='AASVD Acc', color='#55A868')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('AASVD 4-Model Quality & Performance Evaluation (SST-2)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_models, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1:
        h = rect.get_height()
        if h > 0: ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        if h > 0: ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plot_path1 = plots_dir / "aasvd_4models_accuracy_comparison.png"
    plt.savefig(plot_path1, dpi=300)
    plt.close()

    # Chart 2: Pareto Frontier (Accuracy vs Compression Rate)
    fig, ax2 = plt.subplots(figsize=(8, 6))
    for i, model_name in enumerate(plot_models):
        if aasvd_accs[i] > 0:
            ax2.scatter(comp_rates[i], aasvd_accs[i], s=180, label=model_name, zorder=5)
            ax2.annotate(f"{model_name}\n({comp_rates[i]:.2f}x, {aasvd_accs[i]:.1f}%)", (comp_rates[i], aasvd_accs[i]), xytext=(7, -5), textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Physical Compression Rate (x)', fontweight='bold')
    ax2.set_ylabel('AASVD Accuracy (%)', fontweight='bold')
    ax2.set_title('AASVD Efficiency vs. Task Performance Tradeoff (Pareto Frontier)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plot_path2 = plots_dir / "aasvd_pareto_accuracy_vs_compression.png"
    plt.savefig(plot_path2, dpi=300)
    plt.close()
    print(f"Generated and saved comparative visualization charts to:\n- {plot_path1}\n- {plot_path2}", flush=True)
except Exception as e_plot:
    print(f"[WARN] Could not generate comparative plots: {e_plot}", flush=True)

# ==============================================================================
# BATCH QUALITY GATE & DEFERRED EXPORT
# ==============================================================================
print("\n" + "=" * 80, flush=True)
print("STAGE 2: QUALITY-GATED EXPORT TO HUGGING FACE & GITHUB (AASVD)", flush=True)
print("=" * 80, flush=True)

import shutil
try:
    from framework.utils.hf_git_exporter import (
        find_documentation_file,
        upload_to_huggingface,
        update_documentation_table,
        commit_and_push_git,
    )
except ImportError:
    try:
        from Codes.framework.utils.hf_git_exporter import (
            find_documentation_file,
            upload_to_huggingface,
            update_documentation_table,
            commit_and_push_git,
        )
    except ImportError:
        upload_to_huggingface = None
        update_documentation_table = None
        commit_and_push_git = None
        find_documentation_file = None

hf_org = "Hooshaai"
doc_path = find_documentation_file("aasvd") if find_documentation_file else None
pass_count = 0
uploaded_repos = []

for m in models:
    name = m["name"]
    slug = m.get("slug", name)
    out_dir = Path(m["out_dir"])
    weights_dir = out_dir / "checkpoints" / "aasvd"
    hf_export_dir = weights_dir / "hf_export"

    csv_path = Path(m["summary_csv"])
    row = {}
    base_acc = None
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            row_base = df[df["module_name"] == "full"]
            if not row_base.empty:
                col_b = "accuracy_pct" if "accuracy_pct" in row_base.columns else ("mean_accuracy" if "mean_accuracy" in row_base.columns else None)
                if col_b:
                    base_acc = float(row_base[col_b].values[-1])
            row_match = df[df["module_name"] == "aasvd"]
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

    repo_id = f"{hf_org}/svd-linear-attention-{slug}-aasvd"
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
    hf_url = None
    if upload_to_huggingface is not None:
        try:
            hf_url = upload_to_huggingface(
                export_dir=hf_export_dir,
                repo_id=repo_id,
                model_name=slug,
                module_name="aasvd",
                dataset_name="sst2",
                perf_metrics=perf_metrics,
                comp_metrics=comp_metrics,
            )
        except Exception as e_up:
            print(f"  upload_to_huggingface error: {e_up}", flush=True)

    if not hf_url:
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
            api.upload_folder(
                folder_path=str(hf_export_dir),
                repo_id=repo_id,
                commit_message=f"Upload hardened AASVD model weights ({name}) - Acc: {acc:.2f}%",
            )
            hf_url = f"https://huggingface.co/{repo_id}"
        except Exception as e_hf:
            print(f"  Direct HfApi upload error: {e_hf}", flush=True)

    if hf_url:
        print(f"  Successfully uploaded to {hf_url}", flush=True)
        uploaded_repos.append(hf_url)
    else:
        print(f"  Hugging Face upload failed or skipped for {repo_id}", flush=True)

    # Update documentation table
    if update_documentation_table and doc_path and doc_path.exists():
        f1_val = row.get("f1_score", "N/A")
        comp_val = row.get("compression_rate", "1.00")
        svd_layers = row.get("num_compressed_layers", "N/A")
        update_documentation_table(
            doc_path=doc_path,
            model_name=slug,
            module_name="aasvd",
            dataset_name="sst2",
            status=status,
            baseline_acc=base_acc,
            acc=acc,
            f1=f1_val,
            comp_rate=comp_val,
            num_compressed=svd_layers,
            hf_repo_url=hf_url or f"https://huggingface.co/{repo_id}",
        )
        print(f"  Updated documentation table in {doc_path.name}", flush=True)

# Step 4: Clean Git Commit and Push to Dedicated Branch
target_branch = "benchmark-results-aasvd"
print(f"\nCommitting results and documentation updates to dedicated branch '{target_branch}'...", flush=True)
commit_msg = f"feat(benchmark): AASVD evaluation ({pass_count}/{len(models)} passed on SST-2)"
push_res = commit_and_push_git(commit_msg, branch=target_branch) if commit_and_push_git else None

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

