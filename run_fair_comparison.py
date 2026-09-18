import os
import subprocess
import time
import torch

COMPRESSORS = {
    "kmeans": "--modules kmeans --kmeans_clusters 256 --kmeans_lora_r 16",
    "hybrid_svd_layer": "--modules hybrid_svd_layer --hybrid_svd_rank 128 --hybrid_svd_lora_r 16",
    "come": "--modules come --come_sparsity 0.3",
    "sdrm": "--modules sdrm --total_param_ratio 0.7 --lora_r 16",
    "a3": "--modules a3 --a3_max_error 0.5",
    "adaptive_svd_attention": "--modules adaptive_svd_attention"
}

MODELS = {
    "models/distilbert_sst2": "sst2",
    "models/roberta-base": "mnli",
    "models/gpt2": "agnews"
}

os.makedirs("results/fair_comparison", exist_ok=True)

# We enforce 1 worker locally since we run sequentially
print("Starting Fair Comparison Sweep...")

for model_path, dataset in MODELS.items():
    for comp_name, comp_args in COMPRESSORS.items():
        print(f"\n=======================================================")
        print(f"Running {comp_name} on {model_path} ({dataset})")
        print(f"=======================================================")
        
        torch.cuda.empty_cache()
        time.sleep(2) # Give GPU time to flush
        
        cmd = [
            "python", "-u", "Codes/framework/run_quality_sweep.py",
            "--model", model_path,
            "--dataset", dataset,
            "--batch_size", "32",
            "--output_dir", f"results/fair_comparison/{comp_name}_{dataset}",
            "--fresh"
        ]
        
        # Add the custom args
        cmd.extend(comp_args.split())
        
        print(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, timeout=600, check=True)
            print(f"SUCCESS: {comp_name} finished on {dataset}")
        except subprocess.TimeoutExpired:
            print(f"FAILED (TIMEOUT): {comp_name} timed out on {dataset}")
        except subprocess.CalledProcessError as e:
            print(f"FAILED (ERROR): {comp_name} crashed on {dataset} with code {e.returncode}")
        
        torch.cuda.empty_cache()
        time.sleep(2)
        
print("\nAll fair comparison runs complete.")
