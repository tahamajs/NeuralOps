import subprocess
import os
import sys
import time

os.chdir(r"D:\svd_project")

models = [
    {
        "name": "roberta_ablation_r128_s200",
        "cmd": (
            "python -u Codes\\framework\\run_quality_sweep.py "
            "--model models\\roberta-base "
            "--model_family roberta "
            "--dataset sst2 "
            "--dataset_dir Codes\\framework\\data\\glue_data\\sst2 "
            "--output_dir results\\ablation_spectral_recovery\\roberta "
            "--modules adaptive_svd "
            "--batch_size 8 "
            "--rank 128 "
            "--recovery_steps 200 "
            "--timeout 900 "
            "--precision fp16 "
            "--fresh"
        ),
    },
    {
        "name": "gpt2_ablation_r128_s200",
        "cmd": (
            "python -u Codes\\framework\\run_quality_sweep.py "
            "--model models\\gpt2 "
            "--model_family gpt "
            "--dataset sst2 "
            "--dataset_dir Codes\\framework\\data\\glue_data\\sst2 "
            "--output_dir results\\ablation_spectral_recovery\\gpt2 "
            "--modules adaptive_svd "
            "--batch_size 8 "
            "--rank 128 "
            "--recovery_steps 200 "
            "--timeout 900 "
            "--precision fp16 "
            "--parent_checkpoint results\\gpt2_sst2_lora_baseline\\finetuned_model.pt "
            "--fresh"
        ),
    },
]

print("=" * 80, flush=True)
print("STARTING SPECTRAL RECOVERY BOUND ABLATION STUDY (RoBERTa & GPT-2)", flush=True)
print("Ablation Config: svd_rank=128, recovery_steps=200, batch_size=8, precision=fp16", flush=True)
print("=" * 80, flush=True)

for m in models:
    print(f"\n>>> RUNNING ABLATION MODEL: {m['name']}", flush=True)
    t0 = time.time()
    res = subprocess.run(m["cmd"], shell=True)
    elapsed = time.time() - t0
    print(f">>> FINISHED {m['name']} in {elapsed:.1f}s (Exit code: {res.returncode})", flush=True)

print("\n" + "=" * 80, flush=True)
print("ALL ABLATION RUNS COMPLETED SUCCESSFULLY", flush=True)
print("=" * 80, flush=True)
