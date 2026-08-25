import os
import sys
import time
import logging

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

from framework.launch_gpu_server_opt import run_gpu_benchmark_sweep_opt

# Test 20 combos of light modules
# 4 light modules x 3 models (bert, roberta) x 2 datasets (sst2, mrpc) = 16 or 20 combos
models = ["bert", "roberta", "t5"]
datasets = ["sst2", "mrpc"]
light_attentions = ["svd_linear_attention", "adaptive_svd_attention", "asvd_attention", "linformer_attention"]

from framework.eval_all_models_datasets import BenchmarkResult
from framework.launch_gpu_server_opt import _eval_worker_task
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running speedup verification on {device}...")

task_list = []
count = 0
for attn in light_attentions:
    for m in models:
        for d in datasets:
            if count >= 20:
                break
            task_list.append({
                "track": "track1_svd",
                "module_name": attn,
                "module_type": "attention",
                "model_family": m,
                "dataset": d,
                "max_steps": 5,
                "batch_size": 32,
                "seq_len": 128,
                "timeout": 120,
                "device": device,
            })
            count += 1

print(f"Total test combos to run: {len(task_list)}")
start_time = time.time()
timestamps = []

for idx, task in enumerate(task_list, 1):
    t0 = time.time()
    res = _eval_worker_task(task)
    t1 = time.time()
    elapsed = t1 - t0
    timestamps.append((t0, t1, elapsed))
    print(f"[{idx}/{len(task_list)}] {task['module_name']} | {task['model_family']} | {task['dataset']} -> {res.status} ({elapsed:.3f}s)")

total_elapsed = time.time() - start_time
avg_sec = total_elapsed / len(task_list)
print("=" * 60)
print(f"Total Time for {len(task_list)} Combos: {total_elapsed:.3f}s")
print(f"Optimized Avg Seconds per Combo: {avg_sec:.4f}s")
print("=" * 60)
