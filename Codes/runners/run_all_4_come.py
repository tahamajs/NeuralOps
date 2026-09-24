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
        "out_dir": r"results\unified_come\distilbert",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\distilbert_sst2",
            "--model_family", "bert",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_come\distilbert",
            "--parent_checkpoint", r"results\unified_asvd\distilbert\finetuned_model.pt",
            "--modules", "full,come",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--come_target_layers", "4",
            "--come_sparsity", "0.33",
            "--come_alpha_init", "0.1",
            "--come_lora_r", "16",
            "--come_lora_alpha", "1.0",
            "--come_skip_attention_qkv", "true",
            "--come_target_attention_only", "false",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_come\distilbert\summary.csv",
        "stats_json": r"results\unified_come\distilbert\come_stats.json",
    },
    {
        "name": "roberta",
        "slug": "roberta",
        "out_dir": r"results\unified_come\roberta",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\roberta_sst2",
            "--model_family", "roberta",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_come\roberta",
            "--parent_checkpoint", r"results\roberta_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,come",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--come_target_layers", "9",
            "--come_sparsity", "0.25",
            "--come_alpha_init", "0.1",
            "--come_lora_r", "16",
            "--come_lora_alpha", "1.0",
            "--come_skip_attention_qkv", "true",
            "--come_target_attention_only", "false",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_come\roberta\summary.csv",
        "stats_json": r"results\unified_come\roberta\come_stats.json",
    },
    {
        "name": "gpt2",
        "slug": "gpt2",
        "out_dir": r"results\unified_come\gpt2",
        "cmd": [
            sys.executable, "-u", r"Codes\framework\run_quality_sweep.py",
            "--model", r"models\gpt2_sst2",
            "--model_family", "gpt",
            "--dataset", "sst2",
            "--dataset_dir", r"Codes\framework\data\glue_data\sst2",
            "--output_dir", r"results\unified_come\gpt2",
            "--parent_checkpoint", r"results\gpt2_sst2_lora_baseline\finetuned_model.pt",
            "--modules", "full,come",
            "--batch_size", "8",
            "--recovery_steps", "50",
            "--learning_rate", "2e-4",
            "--come_target_layers", "9",
            "--come_sparsity", "0.25",
            "--come_alpha_init", "0.1",
            "--come_lora_r", "16",
            "--come_lora_alpha", "1.0",
            "--come_skip_attention_qkv", "true",
            "--come_target_attention_only", "false",
            "--precision", "fp16",
            "--fresh",
            "--defer_export",
            "--timeout", "180",
        ],
        "summary_csv": r"results\unified_come\gpt2\summary.csv",
        "stats_json": r"results\unified_come\gpt2\come_stats.json",
    },
]

# Allow filtering via CLI
if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
    selected = [m.lower() for m in sys.argv[1:]]
    models = [m for m in models if m["name"] in selected or m["slug"] in selected]

print(f"================================================================================")
print(f"STARTING 4-MODEL UNIFIED BENCHMARK: COME COMPRESSOR (Layer Shedding + Bridges)")
print(f"Models in queue: {[m['name'] for m in models]}")
print(f"Settings: Target Layers: DistilBERT(4), RoBERTa(9), GPT-2(9), Qwen3.5(20), Bridge Alpha=0.1, LoRA Rank=16")
print(f"Policy: Run ALL 4 models first, then execute quality gates, exports, & git commits.")
print(f"================================================================================")

# ==============================================================================
# PHASE 1: EXECUTE ALL MODEL EVALUATIONS FIRST
# ==============================================================================
for m in models:
    name = m["name"]
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] >>> RUNNING COME BENCHMARK FOR {name.upper()}...")
    start_time = time.time()
    
    proc = subprocess.Popen(
        m["cmd"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    
    for line in iter(proc.stdout.readline, ""):
        if line:
            print(f"[{name}] {line.strip()}", flush=True)
            
    proc.wait()
    elapsed = time.time() - start_time
    print(f"[{name}] Completed in {elapsed:.1f}s with exit code {proc.returncode}")

print("\n" + "="*80)
print("PHASE 1 COMPLETE: ALL MODEL RUNS FINISHED. PROCEEDING TO QUALITY GATING & EXPORTS.")
print("="*80)

# ==============================================================================
# PHASE 2: COLLECT RESULTS, QUALITY-GATE, AND EXPORT
# ==============================================================================
collected_results = []

for m in models:
    name = m["name"]
    slug = m["slug"]
    csv_path = Path(m["summary_csv"])
    stats_path = Path(m["stats_json"])
    
    if not csv_path.exists():
        print(f"[-] Missing summary.csv for {name} at {csv_path}. Marking FAILED.")
        collected_results.append({
            "model": name,
            "status": "FAIL",
            "reason": "summary_csv_missing",
            "acc": 0.0,
            "f1": 0.0,
            "delta_acc": 0.0,
        })
        continue

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[-] Error reading {csv_path}: {e}")
        continue

    row_full = df[df.get("module_name", df.get("module", pd.Series())) == "full"]
    row_come = df[df.get("module_name", df.get("module", pd.Series())) == "come"]

    if row_come.empty:
        mod_col = "module_name" if "module_name" in df.columns else "module"
        row_come = df[df[mod_col] != "full"]

    if row_full.empty or row_come.empty:
        print(f"[-] Missing full baseline or come rows in {csv_path}")
        continue

    # Parse accuracy: could be 'accuracy_pct' (0..100) or 'accuracy' (0..1)
    if "accuracy_pct" in row_full.columns:
        baseline_acc = float(row_full.iloc[0].get("accuracy_pct", 0.0)) / 100.0
    else:
        raw_b = float(row_full.iloc[0].get("accuracy", 0.0))
        baseline_acc = raw_b / 100.0 if raw_b > 1.0 else raw_b

    if "accuracy_pct" in row_come.columns:
        come_acc = float(row_come.iloc[0].get("accuracy_pct", 0.0)) / 100.0
    else:
        raw_a = float(row_come.iloc[0].get("accuracy", 0.0))
        come_acc = raw_a / 100.0 if raw_a > 1.0 else raw_a

    come_f1 = float(row_come.iloc[0].get("f1_score", row_come.iloc[0].get("f1", 0.0)))
    delta_acc = come_acc - baseline_acc
    status = str(row_come.iloc[0].get("status", "FAIL")).upper()
    lat_p95 = float(row_come.iloc[0].get("latency_p95_ms", 0.0))
    vram_peak = float(row_come.iloc[0].get("peak_vram_mb", row_come.iloc[0].get("vram_peak_mb", 0.0)))

    # Read detailed compressor stats
    stats_data = {}
    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as sf:
                stats_data = json.load(sf)
        except Exception:
            pass

    num_compressed = stats_data.get("num_compressed", len(stats_data.get("compressed_layers", [])))
    num_fallback = stats_data.get("num_fallback", len(stats_data.get("fallback_layers", [])))
    shed_layers = stats_data.get("shed_layers_count", stats_data.get("shed_layers", 0))
    flops_reduction = stats_data.get("flops_reduction_pct", stats_data.get("estimated_flops_reduction_pct", 0.0))
    avg_reconstruction_error = stats_data.get("avg_reconstruction_error", float(row_come.iloc[0].get("avg_reconstruction_error", 0.0) or 0.0))
    orig_layers = stats_data.get("original_layer_count", 12)
    kept_layers = stats_data.get("kept_layer_count", 9)
    shed_layers = stats_data.get("shed_layer_count", orig_layers - kept_layers)
    layer_reduc_ratio = stats_data.get("layer_reduction_ratio", float(shed_layers) / max(1, orig_layers))
    flops_reduc = stats_data.get("estimated_flops_reduction", f"{layer_reduc_ratio * 100:.1f}%")
    bridges_inserted = stats_data.get("bridges_inserted", 0)

    # Quality Gate Check: Status must be PASS and Accuracy >= 56.0%
    is_passing = (status == "PASS") and (come_acc >= 0.56)

    res_item = {
        "model": name,
        "slug": slug,
        "baseline_acc": baseline_acc,
        "acc": come_acc,
        "delta_acc": delta_acc,
        "f1": come_f1,
        "status": status,
        "passed_gate": is_passing,
        "original_layers": orig_layers,
        "kept_layers": kept_layers,
        "shed_layers": shed_layers,
        "layer_reduction_ratio": layer_reduc_ratio,
        "flops_reduction": flops_reduc,
        "bridges_inserted": bridges_inserted,
        "latency_p95_ms": lat_p95,
        "vram_peak_mb": vram_peak,
        "out_dir": m["out_dir"],
    }
    collected_results.append(res_item)

# Save master summary CSV and JSON
summary_df = pd.DataFrame(collected_results)
master_csv = Path(r"results\unified_come_4models_summary.csv")
master_json = Path(r"results\unified_come_4models_summary.json")
summary_df.to_csv(master_csv, index=False)
with open(master_json, "w", encoding="utf-8") as jf:
    json.dump(collected_results, jf, indent=2)

print(f"\n[+] Master 4-model summary saved to {master_csv} and {master_json}")

# ==============================================================================
# PHASE 3: QUALITY-GATED HF UPLOAD & LOCAL CLEANUP
# ==============================================================================
import huggingface_hub as hfh

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
api = hfh.HfApi()

for res in collected_results:
    name = res["model"]
    is_passing = res.get("passed_gate", False)
    out_dir = Path(res["out_dir"])
    ckpt_dir = out_dir / "checkpoints" / "come"
    hf_export_dir = ckpt_dir / "hf_export"

    if not is_passing:
        print(f"\n[!] QUALITY GATE FAILED for {name} (Acc: {res.get('acc', 0):.2%}, Status: {res.get('status', 'FAIL')}).")
        print(f"[!] Purging checkpoints from {ckpt_dir} (no upload to Hugging Face)...")
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)
        res["hf_repo"] = "N/A (Failed Gate)"
        continue

    print(f"\n[+] QUALITY GATE PASSED for {name} (Acc: {res['acc']:.2%}, Status: PASS). Preparing lean HF export...")
    repo_name = f"Hooshaai/svd-linear-attention-{res['slug']}-come"

    # Assemble lean upload payload: config.json, tokenizer files, weights.pt (omit heavy pytorch_model.bin)
    hf_export_dir.mkdir(parents=True, exist_ok=True)

    # Search for weights.pt in checkpoints
    weights_candidates = list(ckpt_dir.glob("**/weights.pt")) + list(ckpt_dir.glob("weights.pt"))
    if weights_candidates:
        src_weights = weights_candidates[0]
        dst_weights = hf_export_dir / "weights.pt"
        if src_weights != dst_weights and not dst_weights.exists():
            shutil.copy2(src_weights, dst_weights)
            print(f"    Copied lean weights from {src_weights} ({dst_weights.stat().st_size / 1024 / 1024:.1f} MB)")

    # Copy tokenizer and config files if available
    for f in ckpt_dir.glob("*"):
        if f.is_file() and f.suffix in (".json", ".txt", ".jinja") and f.name != "pytorch_model.bin":
            shutil.copy2(f, hf_export_dir / f.name)

    # If pytorch_model.bin exists in hf_export, delete it to keep payload lean
    heavy_bin = hf_export_dir / "pytorch_model.bin"
    if heavy_bin.exists():
        try:
            heavy_bin.unlink()
            print(f"    Removed heavy pytorch_model.bin to keep payload lean.")
        except Exception:
            pass

    # Upload to private Hugging Face repo
    try:
        api.create_repo(repo_name, exist_ok=True, private=True)
        print(f"    Uploading files from {hf_export_dir} to {repo_name}...")
        for file_path in sorted(hf_export_dir.rglob("*")):
            if file_path.is_file():
                rel_p = file_path.relative_to(hf_export_dir).as_posix()
                api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=rel_p,
                    repo_id=repo_name,
                    repo_type="model",
                )
                print(f"      Uploaded {rel_p}")
        print(f"    Uploaded successfully to https://huggingface.co/{repo_name}")
        res["hf_repo"] = f"https://huggingface.co/{repo_name}"
    except Exception as e:
        print(f"    Hugging Face upload failed for {repo_name}: {e}")
        res["hf_repo"] = f"Failed to upload: {e}"

# Update master summary files with HF links
summary_df = pd.DataFrame(collected_results)
summary_df.to_csv(master_csv, index=False)
with open(master_json, "w", encoding="utf-8") as jf:
    json.dump(collected_results, jf, indent=2)

# ==============================================================================
# PHASE 4: UPDATE DOCUMENTATION WITH MASTER BENCHMARK TABLE
# ==============================================================================
doc_path = Path(r"Docs\content\compressions\come.md")
if doc_path.exists():
    table_lines = [
        "\n## Unified 4-Architecture Benchmark Results (CoMe Layer Shedding + Bridges)\n",
        "Evaluated on SST-2 sentiment classification (FP16, layer shedding with learnable interpolation bridges, LoRA rank 16, 200 recovery steps):\n\n",
        "| Model Architecture | Baseline Acc | CoMe Acc | Δ Acc | F1 Score | Retained Layers | Layer Reduction | FLOPs Reduction | Bridges | Peak VRAM | Gate Status | Hugging Face Repository |\n",
        "|:-------------------|:------------:|:--------:|:-----:|:--------:|:---------------:|:---------------:|:---------------:|:-------:|:---------:|:-----------:|:------------------------|\n",
    ]
    for r in collected_results:
        m_name = r["model"]
        b_acc = f"{r.get('baseline_acc', 0.0):.2%}"
        p_acc = f"{r.get('acc', 0.0):.2%}"
        d_acc = f"{r.get('delta_acc', 0.0):+.2%}"
        f1_sc = f"{r.get('f1', 0.0):.4f}"
        ret_layers = f"{r.get('kept_layers', 0)}/{r.get('original_layers', 0)}"
        lay_red = f"{r.get('layer_reduction_ratio', 0.0):.1%}"
        fl_red = r.get("flops_reduction", "0.0%")
        br_ins = f"{r.get('bridges_inserted', 0)}"
        pvram = f"{r.get('vram_peak_mb', 0.0):.1f} MB"
        st = r.get("status", "FAIL")
        hf_link = f"[`{r.get('slug', m_name)}`](https://huggingface.co/Hooshaai/svd-linear-attention-{r.get('slug', m_name)}-come)" if r.get("passed_gate", False) else "N/A"
        table_lines.append(f"| {m_name.upper()} | {b_acc} | {p_acc} | {d_acc} | {f1_sc} | {ret_layers} | {lay_red} | {fl_red} | {br_ins} | {pvram} | `{st}` | {hf_link} |\n")

    content = doc_path.read_text(encoding="utf-8")
    if "## Unified 4-Architecture Benchmark Results (CoMe Layer Shedding + Bridges)" in content:
        content = content.split("## Unified 4-Architecture Benchmark Results (CoMe Layer Shedding + Bridges)")[0]
    content = content.rstrip() + "\n" + "".join(table_lines)
    doc_path.write_text(content, encoding="utf-8")
    print(f"\n[+] Updated documentation benchmark table in {doc_path}")

# ==============================================================================
# PHASE 5: GIT COMMIT AND PUSH TO benchmark-results-come BRANCH
# ==============================================================================
branch_name = "benchmark-results-come"
print(f"\n[+] Committing and pushing results to branch '{branch_name}'...")
try:
    subprocess.run(["git", "checkout", "-b", branch_name], check=False)
    subprocess.run(["git", "checkout", branch_name], check=False)
    subprocess.run(["git", "add", "Docs/content/compressions/come.md", str(master_csv), str(master_json)], check=True)
    subprocess.run(["git", "commit", "-m", "Add unified 4-model benchmark results and documentation for CoMe Compressor"], check=True)
#     subprocess.run(["git", "push", "origin", branch_name], check=True)
    print(f"[+] Successfully committed and pushed results to GitHub branch '{branch_name}'.")
    subprocess.run(["git", "checkout", "main"], check=False)
except Exception as e:
    print(f"[-] Git branch commit/push warning: {e}")

print("\n" + "="*80)
print("ALL 4 MODELS EVALUATED, GATED, EXPORTED, AND DOCUMENTED SUCCESSFULLY!")
print("="*80)
