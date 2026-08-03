#!/usr/bin/env python3
"""
Kaggle GPU Automated Remote Runner
==================================
Packages your framework code, uploads it to Kaggle Datasets, launches a Kaggle GPU Kernel (T4/P100),
monitors the execution status in real-time, and downloads the trained model & results back to your machine.

Usage:
    python push_to_kaggle_gpu.py --command "train --attn_type full --task sst2 --epochs 3"
    python push_to_kaggle_gpu.py --command "sweep --attn_types full nha lsh --seeds 42 123 --task sst2"
    python push_to_kaggle_gpu.py --status
"""

import argparse
import json
import os
import shutil
import sys
import time
import subprocess
from pathlib import Path

USERNAME = "tahamaj"
DATASET_SLUG = f"{USERNAME}/efficient-attention-code"
KERNEL_SLUG = f"{USERNAME}/efficient-attention-gpu-runner"

FRAMEWORK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FRAMEWORK_DIR.parent.parent
BUILD_DIR = FRAMEWORK_DIR / "kaggle_deploy_temp"


def prepare_code_dataset():
    """Package Codes/framework into a clean folder for Kaggle Dataset upload."""
    print("📦 Step 1: Packaging framework codebase...")
    ds_dir = BUILD_DIR / "dataset"
    if ds_dir.exists():
        shutil.rmtree(ds_dir)
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Copy files while ignoring large temp/cache dirs
    ignore = shutil.ignore_patterns(
        "__pycache__", ".git", ".venv", "venv", "results", "logs", "kaggle_deploy_temp", "*.pyc", ".DS_Store"
    )
    shutil.copytree(FRAMEWORK_DIR, ds_dir / "framework", ignore=ignore)

    # dataset-metadata.json
    meta = {
        "title": "Efficient Attention Codebase",
        "id": DATASET_SLUG,
        "licenses": [{"name": "CC0-1.0"}]
    }
    with open(ds_dir / "dataset-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Upload or update Kaggle Dataset
    print(f"☁️ Uploading dataset to Kaggle ({DATASET_SLUG})...")
    res = subprocess.run(["kaggle", "datasets", "status", DATASET_SLUG], capture_output=True, text=True)
    if res.returncode == 0:
        cmd = ["kaggle", "datasets", "version", "-p", str(ds_dir), "-m", "Updated framework codebase"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(ds_dir)]

    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0 and "already exists" not in res.stderr:
        print(f"Warning/Notice on dataset upload: {res.stderr}")
    print("✓ Dataset uploaded/updated successfully")


def prepare_kernel(command_str: str):
    """Generate Jupyter notebook kernel and metadata configured for Kaggle GPU."""
    print("⚡ Step 2: Preparing Kaggle GPU Notebook Kernel...")
    kernel_dir = BUILD_DIR / "kernel"
    if kernel_dir.exists():
        shutil.rmtree(kernel_dir)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    # Notebook cells
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# ⚡ Kaggle GPU Pipeline Execution\n", f"Running command: `python run_kaggle.py {command_str}`"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Install missing requirements\n",
                    "!pip install -q datasets transformers accelerate evaluate scipy scikit-learn deepdiff matplotlib\n",
                    "import os, sys, shutil, torch\n",
                    "print(f'GPU Available: {torch.cuda.is_available()}')\n",
                    "if torch.cuda.is_available():\n",
                    "    print(f'GPU Device: {torch.cuda.get_device_name(0)}')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Copy framework from input dataset to working directory\n",
                    "ds_path = '/kaggle/input/efficient-attention-code/framework'\n",
                    "if not os.path.exists(ds_path):\n",
                    "    ds_path = '/kaggle/input/efficient-attention-code'\n",
                    "work_dir = '/kaggle/working/framework'\n",
                    "if os.path.exists(work_dir): shutil.rmtree(work_dir)\n",
                    "shutil.copytree(ds_path, work_dir)\n",
                    "%cd /kaggle/working/framework\n",
                    "print('Current directory:', os.getcwd())\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Execute pipeline command\n",
                    f"!python run_kaggle.py {command_str}\n"
                ]
            }
        ],
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(kernel_dir / "gpu_run.ipynb", "w") as f:
        json.dump(nb, f, indent=2)

    # kernel-metadata.json
    kmeta = {
        "id": KERNEL_SLUG,
        "title": "Efficient Attention GPU Runner",
        "code_file": "gpu_run.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        "dataset_sources": [DATASET_SLUG],
        "competition_sources": [],
        "kernel_sources": []
    }
    with open(kernel_dir / "kernel-metadata.json", "w") as f:
        json.dump(kmeta, f, indent=2)

    return kernel_dir


def push_and_monitor(kernel_dir: Path):
    """Push kernel to Kaggle GPU and monitor execution progress."""
    print(f"🚀 Step 3: Pushing job to Kaggle GPU ({KERNEL_SLUG})...")
    res = subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(f"❌ Kernel push error: {res.stderr}")
        return False

    print("✅ Kernel pushed! Kaggle GPU worker assigned.")
    print("⏳ Step 4: Monitoring GPU execution status...")

    start_time = time.time()
    while True:
        res = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
        status_line = res.stdout.strip()
        elapsed = int(time.time() - start_time)
        print(f"[{elapsed}s] {status_line}")

        if "has status \"complete\"" in status_line:
            print("🎉 GPU Execution Finished Successfully!")
            break
        elif "has status \"error\"" in status_line:
            print("❌ GPU Execution Failed. Fetching output log...")
            fetch_results()
            return False
        elif "has status \"cancel" in status_line:
            print("⚠️ GPU Execution Cancelled.")
            return False

        time.sleep(15)

    fetch_results()
    return True


def fetch_results():
    """Download output artifacts from Kaggle GPU back to local machine."""
    print("📥 Step 5: Downloading GPU results & models...")
    out_dir = FRAMEWORK_DIR / "results_kaggle_gpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    res = subprocess.run(["kaggle", "kernels", "output", KERNEL_SLUG, "-p", str(out_dir)], capture_output=True, text=True)
    print(res.stdout)
    print(f"✅ Results downloaded to: {out_dir}")


def check_status():
    """Check current status of the Kaggle GPU kernel."""
    res = subprocess.run(["kaggle", "kernels", "status", KERNEL_SLUG], capture_output=True, text=True)
    print(res.stdout)


def main():
    parser = argparse.ArgumentParser(description="Kaggle GPU Remote Pipeline Runner")
    parser.add_argument("--command", default="train --attn_type full --task sst2 --epochs 3", help="Pipeline command string")
    parser.add_argument("--status", action="store_true", help="Check status of running Kaggle GPU kernel")
    parser.add_argument("--fetch", action="store_true", help="Download output files from Kaggle")

    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.fetch:
        fetch_results()
    else:
        prepare_code_dataset()
        kernel_dir = prepare_kernel(args.command)
        push_and_monitor(kernel_dir)


if __name__ == "__main__":
    main()
