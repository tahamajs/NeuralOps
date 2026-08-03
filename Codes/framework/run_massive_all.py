#!/usr/bin/env python3
import os
import sys
import subprocess
import run_kaggle

def main():
    print("Setting up Kaggle environment for massive 10-hour run...")
    run_kaggle.setup_kaggle_environment()
    
    # All 9 GLUE tasks
    tasks = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]
    
    # Extract all 109 registered attention types
    import main as pipeline_main
    fw = pipeline_main._import_framework()
    attn_keys = list(fw["ATTENTION_REGISTRY"].keys())
    
    print(f"Found {len(attn_keys)} attention types. Initiating full sweep over {len(tasks)} tasks.")
    
    for task in tasks:
        cmd = [
            sys.executable, "main.py", "sweep",
            "--attn_types", *attn_keys,
            "--task", task,
            "--epochs", "3",
            "--batch_size", "32",
            "--seeds", "42",
            "--output_dir", "/kaggle/working/results",
            "--log_dir", "/kaggle/working/logs"
        ]
        
        print("\n" + "="*80)
        print(f"🚀 KICKING OFF MASSIVE SWEEP FOR TASK: {task.upper()}")
        print("="*80)
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Sweep for task {task} encountered an error: {e}. Continuing to next task...")

if __name__ == "__main__":
    main()
