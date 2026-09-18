import subprocess
import sys
import os

modules = [
    "none",
    "performer_attention",
    "samba_m3",
    "mamba",
    "hla_inject",
    "gated_deltanet_block",
    "galerkin",
    "phantom",
    "jamba",
    "unified_rope"
]

tasks = ["listops", "imdb"]

def run_lra():
    # we skip cosformer as it's already run/running
    for module in modules:
        for task in tasks:
            print(f"=== Running LRA {task} with {module} ===")
            out_dir = f"D:/SVD_Project/results/lra_{task}_{module}"
            cmd = [
                "python", "-u", "D:/SVD_Project/Codes/framework/train_lra.py",
                "--task", task,
                "--module", module,
                "--output_dir", out_dir
            ]
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"Error running {module} on {task}: {e}")

if __name__ == "__main__":
    run_lra()
