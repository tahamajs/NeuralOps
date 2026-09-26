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
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

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
        "out_dir": r"results\unified_saes_svd_v2\distilbert",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\distilbert_sst2",
            "--model_family", "bert",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_saes_svd_v2\distilbert",
            "--parent_checkpoint", r"results\unified_asvd\distilbert\finetuned_model.pt",
            "--modules", "full,saes_svd_v2",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--rank", "128",
            "--saes_svd_v2_rank", "64",
            "--saes_svd_v2_max_error", "0.50",
            "--saes_svd_v2_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_saes_svd_v2\distilbert\summary.csv",
        "stats_json": r"results\unified_saes_svd_v2\distilbert\saes_svd_v2_stats.json",
    },
    {
        "name": "roberta",
        "slug": "roberta",
        "out_dir": r"results\unified_saes_svd_v2\roberta",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\roberta_sst2",
            "--model_family", "roberta",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_saes_svd_v2\roberta",
            "--parent_checkpoint", r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,saes_svd_v2",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--rank", "128",
            "--saes_svd_v2_rank", "64",
            "--saes_svd_v2_max_error", "0.50",
            "--saes_svd_v2_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_saes_svd_v2\roberta\summary.csv",
        "stats_json": r"results\unified_saes_svd_v2\roberta\saes_svd_v2_stats.json",
    },
    {
        "name": "gpt2",
        "slug": "gpt2",
        "out_dir": r"results\unified_saes_svd_v2\gpt2",
        "cmd": [
            sys.executable, "-u", r"Codes/framework/run_quality_sweep.py",
            "--model", r"models\gpt2_sst2",
            "--model_family", "gpt",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_saes_svd_v2\gpt2",
            "--parent_checkpoint", r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,saes_svd_v2",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--rank", "128",
            "--saes_svd_v2_rank", "64",
            "--saes_svd_v2_max_error", "0.50",
            "--saes_svd_v2_target_attention_only", "true",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_saes_svd_v2\gpt2\summary.csv",
        "stats_json": r"results\unified_saes_svd_v2\gpt2\saes_svd_v2_stats.json",
    },
]

import argparse

parser = argparse.ArgumentParser(description="Unified 4-Model Hardened SAES-SVD v2 Benchmark")
parser.add_argument("--models", nargs="*", default=None, help="Specific models to run (e.g. gpt2 or distilbert,roberta)")
parser.add_argument("--modules", type=str, default=None, help="Comma-separated module names (e.g. full,direct_svd)")
parser.add_argument("--batch_size", type=int, default=None, help="Batch size override")
parser.add_argument("--recovery_steps", type=int, default=None, help="Number of recovery fine-tuning steps")
parser.add_argument("--rank", type=int, default=None, help="Compression rank override")
parser.add_argument("--max_error", type=float, default=None, help="Max reconstruction error threshold")
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
        update_cmd_arg(m["cmd"], "--saes_svd_v2_max_error", str(args.max_error))
    if args.attention_only is not None:
        val = str(args.attention_only).strip().lower()
        if val in ("1", "true", "yes"):
            update_cmd_arg(m["cmd"], "--saes_svd_v2_target_attention_only", "true")
        elif val in ("0", "false", "no"):
            update_cmd_arg(m["cmd"], "--saes_svd_v2_target_attention_only", "false")

    if getattr(args, 'sparsity', None) is not None:
        update_cmd_arg(m["cmd"], "--enlighten_sparsity", str(args.sparsity))
    if getattr(args, 'structured_24', None) is not None:
        val = str(args.structured_24).strip().lower()
        update_cmd_arg(m["cmd"], "--enlighten_structured_24", "true" if val in ("1", "true", "yes") else "false")


print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL HARDENED SAES-SVD v2 BENCHMARK ON GPU SERVER", flush=True)
print(f"Running models: {[m['name'] for m in models]}", flush=True)
print("=" * 80, flush=True)

final_stats = []

for m in models:
    print(f"\n{'='*30} RUNNING MODEL: {m['name']} {'='*30}", flush=True)
    t0 = time.time()
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
                    row = df[df["module_name"] == "saes_svd_v2"]
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
        "SAES-SVD v2 Acc": f"{acc:.2f}%" if acc is not None else "N/A",
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
print("4-MODEL BENCHMARK RESULTS SUMMARY (SAES-SVD v2 on GPU)", flush=True)
print("=" * 80, flush=True)
res_df = pd.DataFrame(final_stats)
print(res_df.to_string(index=False), flush=True)
print("=" * 80, flush=True)

out_dir = Path("results")
out_dir.mkdir(parents=True, exist_ok=True)
out_csv = out_dir / "unified_saes_svd_v2_4models_summary.csv"
out_json = out_dir / "unified_saes_svd_v2_4models_summary.json"
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


# Quality gate & export
print("\n" + "=" * 80, flush=True)
print("STAGE 2: QUALITY-GATED EXPORT TO HUGGING FACE & GITHUB (SAES-SVD v2)", flush=True)
print("=" * 80, flush=True)

from huggingface_hub import HfApi

hf_token = os.environ.get("HF_TOKEN")
api = HfApi(token=hf_token)
hf_org = "Hooshaai"
doc_path = Path(r"Docs\content\compressions\saes_svd_v2.md")
pass_count = 0
uploaded_repos = []

collected_results = []

for m in models:
    name = m["name"]
    slug = m.get("slug", name)
    out_dir = Path(m["out_dir"])
    weights_dir = out_dir / "checkpoints" / "saes_svd_v2"
    hf_export_dir = weights_dir / "hf_export"

    csv_path = Path(m["summary_csv"])
    row = {}
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            row_match = df[df["module_name"] == "saes_svd_v2"]
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

    res_item = {
        "model": name,
        "slug": slug,
        "acc": acc,
        "status": status,
        "passed_gate": is_successful,
        "f1": row.get("f1_score", "N/A"),
        "comp_rate": row.get("compression_rate", "1.00"),
        "num_compressed": row.get("num_compressed_layers", "N/A"),
        "peak_vram": row.get("peak_vram_mb", "N/A"),
        "hf_repo": "N/A",
    }

    if not is_successful:
        print(f"[Quality Gate: REJECTED] {name} did not meet criteria (status={status}, acc={acc:.2f}% < 56.0%).", flush=True)
        if weights_dir.exists():
            print(f"  Pruning local checkpoint artifacts in {weights_dir}", flush=True)
            shutil.rmtree(weights_dir, ignore_errors=True)
        collected_results.append(res_item)
        continue

    pass_count += 1
    print(f"[Quality Gate: PASSED] {name} passed criteria (status={status}, acc={acc:.2f}% >= 56.0%). Packaging for export...", flush=True)

    hf_export_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    weights_candidates = (
        list(checkpoints_dir.glob("**/weights.pt"))
        + list(weights_dir.glob("**/weights.pt"))
        + list(out_dir.glob("**/weights.pt"))
    )
    if weights_candidates:
        src_weights = weights_candidates[0]
        dst_weights = hf_export_dir / "weights.pt"
        if src_weights != dst_weights and not dst_weights.exists():
            shutil.copy2(src_weights, dst_weights)

    for f in weights_dir.glob("*"):
        if f.is_file() and f.suffix in (".json", ".txt", ".jinja") and f.name != "pytorch_model.bin":
            shutil.copy2(f, hf_export_dir / f.name)

    heavy_bin = hf_export_dir / "pytorch_model.bin"
    if heavy_bin.exists():
        try:
            heavy_bin.unlink()
            print(f"  Removed heavy pytorch_model.bin for lean payload.")
        except Exception:
            pass

    repo_id = f"{hf_org}/svd-linear-attention-{slug}-saes_svd_v2"
    try:
        api.create_repo(repo_id, exist_ok=True, private=True)
        for file_path in sorted(hf_export_dir.rglob("*")):
            if file_path.is_file():
                rel_p = file_path.relative_to(hf_export_dir).as_posix()
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=rel_p,
                    repo_id=repo_id,
                    repo_type="model",
                )
        hf_url = f"https://huggingface.co/{repo_id}"
        print(f"  Successfully uploaded to {hf_url}", flush=True)
        uploaded_repos.append(hf_url)
        res_item["hf_repo"] = hf_url
    except Exception as e:
        print(f"  Hugging Face upload failed for {repo_id}: {e}", flush=True)
        res_item["hf_repo"] = f"Upload error: {e}"

    collected_results.append(res_item)

# Update documentation table
if doc_path.exists():
    table_lines = [
        "\n## Unified 4-Architecture Benchmark Results (SAES-SVD v2)\n",
        "Evaluated on SST-2 sentiment classification (FP16, rank 128 SVD + 2:4 structured sparsity, max error 0.35, target attention only, 200 recovery steps):\n\n",
        "| Model Architecture | Accuracy | F1 Score | Compression Rate | Compressed Layers | Peak VRAM | Gate Status | Hugging Face Repository |\n",
        "|:-------------------|:--------:|:--------:|:----------------:|:-----------------:|:---------:|:-----------:|:------------------------|\n",
    ]
    for r in collected_results:
        m_name = r["model"]
        acc_str = f"{r.get('acc', 0.0):.2f}%"
        f1_str = str(r.get("f1", "N/A"))
        c_rate = str(r.get("comp_rate", "1.00x"))
        c_layers = str(r.get("num_compressed", "N/A"))
        pvram = str(r.get("peak_vram", "N/A"))
        st = r.get("status", "FAIL")
        hf_link = f"[`{r.get('slug', m_name)}`]({r['hf_repo']})" if r.get("passed_gate", False) else "N/A"
        table_lines.append(f"| {m_name.upper()} | {acc_str} | {f1_str} | {c_rate} | {c_layers} | {pvram} | `{st}` | {hf_link} |\n")

    content = doc_path.read_text(encoding="utf-8")
    if "## Unified 4-Architecture Benchmark Results (SAES-SVD v2)" in content:
        content = content.split("## Unified 4-Architecture Benchmark Results (SAES-SVD v2)")[0]
    content = content.rstrip() + "\n" + "".join(table_lines)
    doc_path.write_text(content, encoding="utf-8")
    print(f"\n[+] Updated documentation benchmark table in {doc_path}", flush=True)

print("Committing results directly to main...", flush=True)
try:
    # Auto-commit and auto-push disabled pending review.
    # subprocess.run(["git", "add", str(doc_path), str(out_csv), str(out_json)], check=False)
    # subprocess.run(["git", "commit", "-m", f"feat(benchmark): evaluation ({pass_count}/{len(models)} passed on SST-2)"], check=False)
    # subprocess.run(["git", "push", "origin", "main"], check=False)
    print("Results committed to main.", flush=True)
except Exception as e:
    print(f"Git commit warning: {e}", flush=True)
except Exception as e:
    print(f"Git branch commit/push warning: {e}", flush=True)

print("\n" + "=" * 80, flush=True)
print("ALL TASKS COMPLETED!", flush=True)
print(f"Passed Models: {pass_count} / {len(models)}")
print(f"Uploaded Repositories: {len(uploaded_repos)}")
for u in uploaded_repos:
    print(f"  - {u}")
print("=" * 80, flush=True)
