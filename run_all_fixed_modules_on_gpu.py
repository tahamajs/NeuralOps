import subprocess
import sys
import os

modules = [
    "rala",
    "regla",
    "compressed_mha",
    "eat",
    "nha",
    "adasplash",
    "squeezing_heads",
    "svd_linear_attention"
]

def run_command(cmd):
    print(f"Running: {cmd}", flush=True)
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running {cmd}", flush=True)
    return result.returncode

def main():
    for mod in modules:
        script_name = f"run_all_4_{mod}.py"
        if os.path.exists(script_name):
            print(f"====== Running benchmark for {mod} ======")
            run_command(f"python -u {script_name}")
        else:
            print(f"Script {script_name} not found, skipping...")
            
    print("====== Benchmarks finished, uploading results ======")
    run_command("python -u upload_all_passing_to_hf.py")

if __name__ == "__main__":
    main()
