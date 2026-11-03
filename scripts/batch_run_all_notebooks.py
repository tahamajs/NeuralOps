#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Batch Runner for all canonical notebooks directly on GPU-3090-1.
"""
import os
import glob
import subprocess
import sys
import time

project_dir = os.path.expanduser("~/NeuralOps")
notebooks_dir = os.path.join(project_dir, "notebooks")
output_dir = os.path.join(project_dir, "results", "notebooks_executed")
os.makedirs(output_dir, exist_ok=True)

all_notebooks = sorted([
    f for f in glob.glob(os.path.join(notebooks_dir, "*.ipynb"))
    if not os.path.basename(f).startswith(".")
])

print("Total Notebooks queued for Physical GPU execution: {}".format(len(all_notebooks)), flush=True)
results = []

for idx, nb in enumerate(all_notebooks, 1):
    nb_name = os.path.basename(nb)
    out_nb = os.path.join(output_dir, nb_name)
    print("\n[{}/{}] STARTING ON GPU-3090-1: {}".format(idx, len(all_notebooks), nb_name), flush=True)
    t0 = time.time()
    
    cmd = [
        os.path.expanduser("~/.conda/envs/neuralops_cu113/bin/python"),
        os.path.join(project_dir, "scripts", "run_notebook_gpu.py"),
        nb,
        out_nb
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.path.expanduser("~/.conda/envs/neuralops_cu113/lib:") + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = project_dir + ":" + env.get("PYTHONPATH", "")
    
    proc = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    dt = time.time() - t0
    status = "SUCCESS" if proc.returncode == 0 else "FAILED (code {})".format(proc.returncode)
    print("[{}/{}] {} Finished in {:.1f}s -> {}".format(idx, len(all_notebooks), nb_name, dt, status), flush=True)
    if proc.stdout:
        print(proc.stdout.strip(), flush=True)
    results.append((nb_name, status, dt))

print("\n" + "=" * 80, flush=True)
print("BATCH NOTEBOOK EXECUTION ON GPU-3090-1 COMPLETED", flush=True)
print("=" * 80, flush=True)
for nb_name, status, dt in results:
    print("{:<65} | {:<15} | {:6.1f}s".format(nb_name, status, dt), flush=True)
