import os
import sys
import time
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

from framework.launch_gpu_server_opt import _eval_worker_task

models = ["bert", "roberta", "t5"]
datasets = ["sst2", "mrpc"]
light_attentions = ["svd_linear_attention", "adaptive_svd_attention", "asvd_attention", "linformer_attention"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running profile test on {device} (VRAM Target: ~5.5-6.5GB)...")

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
                "batch_size": 64,
                "seq_len": 256,
                "timeout": 120,
                "device": device,
            })
            count += 1

print(f"Total test combos: {len(task_list)}")
start_time = time.time()
timestamps = []

for idx, task in enumerate(task_list, 1):
    t0 = time.time()
    res = _eval_worker_task(task)
    t1 = time.time()
    elapsed = t1 - t0
    timestamps.append((t0, t1, elapsed))
    print(f"[{idx}/{len(task_list)}] {task['module_name']} | {task['model_family']} | {task['dataset']} -> {res.status} (fwd: {res.forward_latency_ms:.2f}ms, bwd: {res.backward_latency_ms:.2f}ms, wall: {elapsed:.3f}s)")

total_elapsed = time.time() - start_time
avg_sec = total_elapsed / len(task_list)
print("=" * 60)
print(f"Total Wall-Clock Time: {total_elapsed:.3f}s")
print(f"Overall Avg Sec/Combo: {avg_sec:.4f}s")
if len(timestamps) > 6:
    warm_elapsed = sum(e for _, _, e in timestamps[6:])
    warm_avg = warm_elapsed / len(timestamps[6:])
    print(f"Warm/Cached Avg Sec/Combo (Combos 7-20): {warm_avg:.4f}s")
print("=" * 60)
