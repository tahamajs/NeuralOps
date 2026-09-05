import subprocess
import os
import sys
import time
import json
import pandas as pd

os.chdir(r"D:\svd_project")

models = [
    {
        "name": "DistilBERT",
        "cmd": (
            r'"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py '
            r"--model models\distilbert_sst2 "
            r"--model_family bert "
            r"--dataset sst2 "
            r"--dataset_dir Codes\framework\data\glue_data\sst2 "
            r"--output_dir results\unified_a3\distilbert "
            r"--modules a3 "
            r"--batch_size 8 "
            r"--recovery_steps 100 "
            r"--timeout 1200 "
            r"--precision fp16 "
            r"--fresh"
        ),
        "summary": r"D:\SVD_Project\results\unified_a3\distilbert\summary.csv",
    },
    {
        "name": "RoBERTa",
        "cmd": (
            r'"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py '
            r"--model models\roberta-base "
            r"--model_family roberta "
            r"--dataset sst2 "
            r"--dataset_dir Codes\framework\data\glue_data\sst2 "
            r"--output_dir results\unified_a3\roberta "
            r"--modules a3 "
            r"--batch_size 8 "
            r"--recovery_steps 100 "
            r"--timeout 1200 "
            r"--precision fp16 "
            r"--parent_checkpoint results\roberta_sst2_lora_baseline\finetuned_model.pt "
            r"--fresh"
        ),
        "summary": r"D:\SVD_Project\results\unified_a3\roberta\summary.csv",
    },
    {
        "name": "GPT-2",
        "cmd": (
            r'"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py '
            r"--model models\gpt2 "
            r"--model_family gpt "
            r"--dataset sst2 "
            r"--dataset_dir Codes\framework\data\glue_data\sst2 "
            r"--output_dir results\unified_a3\gpt2 "
            r"--modules a3 "
            r"--batch_size 8 "
            r"--recovery_steps 100 "
            r"--timeout 1200 "
            r"--precision fp16 "
            r"--parent_checkpoint results\gpt2_sst2_lora_baseline\finetuned_model.pt "
            r"--fresh"
        ),
        "summary": r"D:\SVD_Project\results\unified_a3\gpt2\summary.csv",
    },
    {
        "name": "Qwen3.5",
        "cmd": (
            r'"C:\Users\Hadi Schumacher\anaconda3\python.exe" -u Codes\framework\run_quality_sweep.py '
            r"--model models\Qwen3.5-0.8B "
            r"--model_family qwen3_5 "
            r"--dataset sst2 "
            r"--dataset_dir Codes\framework\data\glue_data\sst2 "
            r"--output_dir results\unified_a3\qwen3_5 "
            r"--modules a3 "
            r"--batch_size 8 "
            r"--recovery_steps 100 "
            r"--timeout 1200 "
            r"--precision fp16 "
            r"--fresh"
        ),
        "summary": r"D:\SVD_Project\results\unified_a3\qwen3_5\summary.csv",
    },
]

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL A3 COMPRESSOR GPU BENCHMARK SWEEP", flush=True)
print("Standard Config: module=a3, dataset=sst2, recovery_steps=100, batch_size=8, precision=fp16", flush=True)
print("=" * 80, flush=True)

results_summary = []

for m in models:
    print(f"\n==================================================================", flush=True)
    print(f">>> [RUNNING MODEL] {m['name']}", flush=True)
    print(f"==================================================================", flush=True)
    t0 = time.time()
    res = subprocess.run(m["cmd"], shell=True)
    elapsed = time.time() - t0
    print(f"\n>>> [FINISHED] {m['name']} in {elapsed:.1f}s (Exit code: {res.returncode})", flush=True)
    
    # Read summary
    status = "ERROR"
    acc = 0.0
    threshold = 56.0
    if os.path.exists(m["summary"]):
        try:
            df = pd.read_csv(m["summary"])
            row = df[df["module_name"] == "a3"]
            if not row.empty:
                status = row["status"].values[0]
                acc = float(row["accuracy_pct"].values[0])
                if "threshold_pct" in row:
                    threshold = float(row["threshold_pct"].values[0])
        except Exception as e:
            print(f"Error parsing {m['summary']}: {e}", flush=True)
            
    results_summary.append({
        "Model": m["name"],
        "Module": "a3",
        "Status": status,
        "Accuracy (%)": acc,
        "Threshold (%)": threshold,
        "Elapsed (s)": round(elapsed, 1),
    })

print("\n" + "=" * 80, flush=True)
print("FINAL 4-MODEL A3 GPU BENCHMARK RESULTS TABLE", flush=True)
print("=" * 80, flush=True)
res_df = pd.DataFrame(results_summary)
print(res_df.to_markdown(index=False) if hasattr(res_df, "to_markdown") else res_df.to_string(index=False), flush=True)

# Save consolidated results to JSON and CSV
out_csv = r"D:\SVD_Project\results\unified_a3_4models_summary.csv"
out_json = r"D:\SVD_Project\results\unified_a3_4models_summary.json"
res_df.to_csv(out_csv, index=False)
with open(out_json, "w") as f:
    json.dump(results_summary, f, indent=2)
print(f"\nSaved consolidated summaries to:\n- {out_csv}\n- {out_json}", flush=True)
