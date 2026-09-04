import subprocess
import os
import sys
import time

os.chdir(r"D:\svd_project")

models = [
    {
        "name": "distilbert",
        "cmd": (
            "python -u Codes\\framework\\run_quality_sweep.py "
            "--model models\\distilbert_sst2 "
            "--model_family bert "
            "--dataset sst2 "
            "--dataset_dir Codes\\framework\\data\\glue_data\\sst2 "
            "--output_dir results\\unified_adaptive_svd\\distilbert "
            "--modules adaptive_svd "
            "--batch_size 8 "
            "--recovery_steps 50 "
            "--timeout 600 "
            "--precision fp16 "
            "--fresh"
        ),
    },
    {
        "name": "roberta",
        "cmd": (
            "python -u Codes\\framework\\run_quality_sweep.py "
            "--model models\\roberta-base "
            "--model_family roberta "
            "--dataset sst2 "
            "--dataset_dir Codes\\framework\\data\\glue_data\\sst2 "
            "--output_dir results\\unified_adaptive_svd\\roberta "
            "--modules adaptive_svd "
            "--batch_size 8 "
            "--recovery_steps 50 "
            "--timeout 600 "
            "--precision fp16 "
            "--fresh"
        ),
    },
    {
        "name": "gpt2",
        "cmd": (
            "python -u Codes\\framework\\run_quality_sweep.py "
            "--model models\\gpt2 "
            "--model_family gpt "
            "--dataset sst2 "
            "--dataset_dir Codes\\framework\\data\\glue_data\\sst2 "
            "--output_dir results\\unified_adaptive_svd\\gpt2 "
            "--modules adaptive_svd "
            "--batch_size 8 "
            "--recovery_steps 50 "
            "--timeout 600 "
            "--precision fp16 "
            "--parent_checkpoint results\\gpt2_sst2_lora_baseline\\finetuned_model.pt "
            "--fresh"
        ),
    },
    {
        "name": "qwen3_5",
        "cmd": (
            "python -u Codes\\framework\\run_quality_sweep.py "
            "--model models\\Qwen3.5-0.8B "
            "--model_family qwen3_5 "
            "--dataset sst2 "
            "--dataset_dir Codes\\framework\\data\\glue_data\\sst2 "
            "--output_dir results\\unified_adaptive_svd\\qwen3_5 "
            "--modules adaptive_svd "
            "--batch_size 8 "
            "--recovery_steps 50 "
            "--timeout 600 "
            "--precision fp16 "
            "--fresh"
        ),
    },
]

print("=" * 80, flush=True)
print("STARTING UNIFIED 4-MODEL ADAPTIVE SVD BENCHMARK RUN", flush=True)
print("Standard Config: dataset=sst2, recovery_steps=50, batch_size=8, precision=fp16", flush=True)
print("=" * 80, flush=True)

for m in models:
    print(f"\n>>> RUNNING MODEL: {m['name']}", flush=True)
    t0 = time.time()
    res = subprocess.run(m["cmd"], shell=True)
    elapsed = time.time() - t0
    print(f">>> FINISHED {m['name']} in {elapsed:.1f}s (Exit code: {res.returncode})", flush=True)

print("\n" + "=" * 80, flush=True)
print("ALL 4 MODELS COMPLETED IN UNIFIED BENCHMARK (ADAPTIVE SVD)", flush=True)
print("=" * 80, flush=True)
