import subprocess
import os
import sys
import time

os.chdir(r"D:\svd_project")

cmd = (
    "python -u Codes\\framework\\run_quality_sweep.py "
    "--model models\\gpt2 "
    "--model_family gpt "
    "--dataset sst2 "
    "--dataset_dir Codes\\framework\\data\\glue_data\\sst2 "
    "--output_dir results\\unified_asvd\\gpt2 "
    "--modules asvd "
    "--batch_size 8 "
    "--recovery_steps 200 "
    "--learning_rate 1e-4 "
    "--lora_rank 32 "
    "--asvd_dynamic_rank "
    "--asvd_energy_threshold 0.95 "
    "--asvd_error_threshold 0.20 "
    "--timeout 1500 "
    "--precision fp16 "
    "--parent_checkpoint results\\gpt2_sst2_lora_baseline\\finetuned_model.pt "
    "--fresh"
)

print("Running GPT-2 with frozen base SVD weights + lr=1e-4 + warmup...", flush=True)
res = subprocess.run(cmd, shell=True)
print(f"Finished with exit code: {res.returncode}", flush=True)
