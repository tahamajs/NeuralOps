import subprocess, os, sys, time, json, shutil
import pandas as pd
from pathlib import Path

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

if os.path.exists(r"D:\svd_project"):
    os.chdir(r"D:\svd_project")

MODULE = "hgrn2_gated_deltanet"
DESCRIPTION = "HGRN-2 Gated DeltaNet Hybrid Attention Module"

models = [
    {"name": "distilbert", "slug": "distilbert", "out_dir": rf"results\unified_{MODULE}\distilbert",
     "cmd": [sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
             "--model", r"models\distilbert_sst2", "--model_family", "bert", "--dataset", "sst2",
             "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
             "--output_dir", rf"results\unified_{MODULE}\distilbert",
             "--parent_checkpoint", r"results\unified_asvd\distilbert\finetuned_model.pt",
             "--modules", f"full,{MODULE}", "--batch_size", "8", "--recovery_steps", "200",
             "--learning_rate", "2e-4", "--lora_rank", "16", "--rank", "32",
             "--precision", "fp16", "--fresh", "--defer_export", "--timeout", "1200"],
     "summary_csv": rf"results\unified_{MODULE}\distilbert\summary.csv",
     "stats_json": rf"results\unified_{MODULE}\distilbert\{MODULE}_stats.json"},
    {"name": "roberta", "slug": "roberta", "out_dir": rf"results\unified_{MODULE}\roberta",
     "cmd": [sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
             "--model", r"models\roberta-base", "--model_family", "roberta", "--dataset", "sst2",
             "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
             "--output_dir", rf"results\unified_{MODULE}\roberta",
             "--parent_checkpoint", r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
             "--modules", f"full,{MODULE}", "--batch_size", "8", "--recovery_steps", "200",
             "--learning_rate", "2e-4", "--lora_rank", "16", "--rank", "32",
             "--precision", "fp16", "--fresh", "--defer_export", "--timeout", "1200"],
     "summary_csv": rf"results\unified_{MODULE}\roberta\summary.csv",
     "stats_json": rf"results\unified_{MODULE}\roberta\{MODULE}_stats.json"},
    {"name": "gpt2", "slug": "gpt2", "out_dir": rf"results\unified_{MODULE}\gpt2",
     "cmd": [sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
             "--model", r"models\gpt2", "--model_family", "gpt", "--dataset", "sst2",
             "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
             "--output_dir", rf"results\unified_{MODULE}\gpt2",
             "--parent_checkpoint", r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
             "--modules", f"full,{MODULE}", "--batch_size", "8", "--recovery_steps", "200",
             "--learning_rate", "2e-4", "--lora_rank", "16", "--rank", "32",
             "--precision", "fp16", "--fresh", "--defer_export", "--timeout", "1200"],
     "summary_csv": rf"results\unified_{MODULE}\gpt2\summary.csv",
     "stats_json": rf"results\unified_{MODULE}\gpt2\{MODULE}_stats.json"},
    {"name": "qwen3_5", "slug": "qwen3_5", "out_dir": rf"results\unified_{MODULE}\qwen3_5",
     "cmd": [sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
             "--model", r"models\Qwen3.5-0.8B", "--model_family", "qwen3_5", "--dataset", "sst2",
             "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
             "--output_dir", rf"results\unified_{MODULE}\qwen3_5",
             "--parent_checkpoint", r"results\unified_asvd\qwen3_5\finetuned_model.pt",
             "--modules", f"full,{MODULE}", "--batch_size", "8", "--recovery_steps", "200",
             "--learning_rate", "2e-4", "--lora_rank", "16", "--rank", "32",
             "--precision", "fp16", "--fresh", "--defer_export", "--timeout", "1500"],
     "summary_csv": rf"results\unified_{MODULE}\qwen3_5\summary.csv",
     "stats_json": rf"results\unified_{MODULE}\qwen3_5\{MODULE}_stats.json"},
]

import argparse
parser = argparse.ArgumentParser(description=f"Unified 4-Model {MODULE.upper()} Benchmark")
parser.add_argument("--models", nargs="*", default=None)
parser.add_argument("--modules", type=str, default=None)
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--recovery_steps", type=int, default=None)
parser.add_argument("--rank", type=int, default=None)
parser.add_argument("--lora_rank", type=int, default=None)
parser.add_argument("positional_models", nargs="*", default=None)
args = parser.parse_args()

selected_models = []
if args.models:
    for item in args.models:
        for sub in item.split(","): selected_models.append(sub.strip()) if sub.strip() else None
elif args.positional_models:
    for item in args.positional_models:
        for sub in item.split(","): selected_models.append(sub.strip()) if sub.strip() else None
if selected_models:
    models = [m for m in models if m["name"] in set(selected_models)]

def update_cmd_arg(cmd, flag, val):
    try: cmd[cmd.index(flag) + 1] = str(val)
    except ValueError: cmd.extend([flag, str(val)])

for m in models:
    if args.modules: update_cmd_arg(m["cmd"], "--modules", args.modules)
    if args.batch_size is not None: update_cmd_arg(m["cmd"], "--batch_size", args.batch_size)
    if args.recovery_steps is not None: update_cmd_arg(m["cmd"], "--recovery_steps", args.recovery_steps)
    if args.rank is not None: update_cmd_arg(m["cmd"], "--rank", args.rank)
    if args.lora_rank is not None: update_cmd_arg(m["cmd"], "--lora_rank", args.lora_rank)

start_time = time.time()
for m in models:
    name, out_dir, cmd = m["name"], Path(m["out_dir"]), m["cmd"]
    print(f"\n{'='*56}\nStarting {name.upper()} Benchmark...\nCommand: {' '.join(cmd)}\n{'='*56}\n", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "console.log", "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush(); lf.write(line); lf.flush()
        proc.wait()
    if proc.returncode != 0:
        print(f"\n[ERROR] Benchmark failed for {name} with returncode {proc.returncode}.", flush=True)
    else:
        print(f"\n[SUCCESS] Finished Benchmark for {name}.", flush=True)

print(f"\nAll Requested Models Processed in {time.time() - start_time:.2f}s\n")
print("\n--- Summary Results ---")
for m in models:
    sp = Path(m["summary_csv"])
    if sp.exists():
        try: df = pd.read_csv(sp); print(f"\nSummary for {m['name']}:\n{df.to_string(index=False)}")
        except Exception as e: print(f"Could not read summary CSV for {m['name']}: {e}")
    else: print(f"No summary CSV found for {m['name']} at {sp}")

print(f"\nStarting Quality Gate & Automated Hugging Face Export...\n", flush=True)
try:
    from huggingface_hub import HfApi; api = HfApi()
except ImportError:
    api = None; print("[WARN] huggingface_hub not installed. HF upload skipped.", flush=True)

hf_org, pass_count, uploaded_repos = "Hooshaai", 0, []
for m in models:
    name, slug, out_dir = m["name"], m["slug"], Path(m["out_dir"])
    summary_path, weights_dir, hf_export_dir = Path(m["summary_csv"]), out_dir / "weights", out_dir / "hf_export"
    if not summary_path.exists():
        print(f"[Quality Gate: REJECTED] {name} summary CSV missing.", flush=True)
        shutil.rmtree(weights_dir, ignore_errors=True); continue
    row = {}
    try:
        df = pd.read_csv(summary_path)
        if not df.empty:
            cand = df[df["module"] != "full"] if "module" in df.columns else df
            row = (cand if not cand.empty else df).iloc[-1].to_dict()
    except Exception as e: print(f"Error reading summary for {name}: {e}", flush=True)
    status = str(row.get("status", "FAIL"))
    try: acc = float(row.get("accuracy_pct", row.get("mean_accuracy", 0.0)))
    except: acc = 0.0
    if not ((status == "PASS") and (acc >= 56.0)):
        print(f"[Quality Gate: REJECTED] {name} status={status}, acc={acc:.2f}%", flush=True)
        shutil.rmtree(weights_dir, ignore_errors=True); continue
    pass_count += 1
    print(f"[Quality Gate: PASSED] {name} acc={acc:.2f}%", flush=True)
    hf_export_dir.mkdir(parents=True, exist_ok=True)
    wcs = (list((out_dir / "checkpoints").glob("**/weights.pt")) +
           list(weights_dir.glob("**/weights.pt")) + list(out_dir.glob("**/weights.pt")))
    if wcs:
        src = wcs[0]; dst = hf_export_dir / "weights.pt"
        if src.resolve() != dst.resolve(): shutil.copy2(src, dst)
        for fx in src.parent.glob("*"):
            if fx.is_file() and fx.name not in ("weights.pt", "pytorch_model.bin"):
                shutil.copy2(fx, hf_export_dir / fx.name)
    with open(hf_export_dir / "metrics.json", "w") as mf:
        json.dump({"model_name": name, "module": MODULE, "status": status, "accuracy_pct": acc,
                   "f1": float(row.get("f1", 0)), "full_metrics": row}, mf, indent=2)
    card = f"""---
tags: [svd-linear-attention, model-compression, {MODULE}, {slug}, pytorch]
datasets: [sst2]
metrics: [accuracy, f1]
---
# SVD Linear Attention: {name.upper()} ({MODULE})
{DESCRIPTION} — validated checkpoint evaluated on SST-2.
- **Accuracy**: `{acc:.2f}%` | **Status**: `{status}` (threshold >= 56%)
- **F1**: `{row.get('f1', 'N/A')}` | **Peak VRAM**: `{row.get('peak_vram_mb', 'N/A')} MB`
```python
import torch; checkpoint = torch.load("weights.pt", map_location="cpu")
```
Part of the SVD Linear Attention Benchmark Suite.
"""
    (hf_export_dir / "README.md").write_text(card)
    repo_id = f"{hf_org}/svd-linear-attention-{slug}-{MODULE}"
    if api:
        try:
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
            api.upload_folder(folder_path=str(hf_export_dir), repo_id=repo_id,
                              commit_message=f"Upload {MODULE.upper()} {name} - Acc: {acc:.2f}%")
            print(f"Uploaded {name} to {repo_id}", flush=True); uploaded_repos.append(repo_id)
        except Exception as e: print(f"Upload failed for {repo_id}: {e}", flush=True)

print(f"\nExport Complete: {pass_count}/{len(models)} passed quality gate.", flush=True)
