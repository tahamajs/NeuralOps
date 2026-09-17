import subprocess, sys, csv
import os

modules = ["compressed_mha", "mla", "asvd", "hamming", "nasc", "galerkin", "dobi", "optimal_subspace_projection"]
ranks = [32, 64, 128, 256]
model = "distilbert"

for mod in modules:
    for rank in ranks:
        cmd = [
            sys.executable, "-u", f"Codes/runners/run_all_4_{mod}.py",
            "--models", model,
            "--recovery_steps", "50",
            "--rank", str(rank)
        ]
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, timeout=300, check=False)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT for {mod} rank {rank}")
        print("-" * 50)
