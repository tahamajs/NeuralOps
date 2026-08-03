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
    res = subprocess.run(["kaggle", "datasets", "list", "--mine"], capture_output=True, text=True)
    if DATASET_SLUG in res.stdout or "efficient-attention-code" in res.stdout:
        cmd = ["kaggle", "datasets", "version", "-p", str(ds_dir), "-m", "Updated framework codebase", "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(ds_dir), "--dir-mode", "zip"]

    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0 and "already exists" not in res.stderr:
        print(f"Notice on dataset upload: {res.stderr}")
    print("✓ Dataset uploaded/updated successfully")
    print("⏳ Waiting 20 seconds for Kaggle to process new dataset version...")
    time.sleep(20)


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
                    "# ── CRITICAL FIX: Install PyTorch 2.1.2 + cu118 (LAST version supporting P100 sm_60) ──\n",
                    "# PyTorch 2.2+ dropped sm_60 support; 2.1.2+cu118 is the last compatible build.\n",
                    "import subprocess, sys\n",
                    "subprocess.check_call([\n",
                    "    sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall',\n",
                    "    'torch==2.1.2', 'torchvision==0.16.2', 'torchaudio==2.1.2',\n",
                    "    '--index-url', 'https://download.pytorch.org/whl/cu118'\n",
                    "])\n",
                    "subprocess.check_call([\n",
                    "    sys.executable, '-m', 'pip', 'install', '-q',\n",
                    "    'datasets', 'transformers', 'accelerate', 'evaluate',\n",
                    "    'scipy', 'scikit-learn', 'deepdiff', 'matplotlib', 'asgiref', 'channels'\n",
                    "])\n",
                    "print('✓ PyTorch 2.1.2+cu118 installed — P100/sm_60 fully supported')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Verify GPU + correct PyTorch version\n",
                    "import subprocess, sys\n",
                    "result = subprocess.run([sys.executable, '-c',\n",
                    "    'import torch; print(f\"PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}\"); '\n",
                    "    'torch.cuda.is_available() and print(f\"GPU: {torch.cuda.get_device_name(0)}\") '\n",
                    "], capture_output=True, text=True)\n",
                    "print(result.stdout)\n",
                    "if result.stderr: print('STDERR:', result.stderr[:500])\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Dynamically extract and locate framework directory in /kaggle/input/\n",
                    "import glob, os, sys, shutil, zipfile\n",
                    "work_dir = '/kaggle/working/framework'\n",
                    "if os.path.exists(work_dir): shutil.rmtree(work_dir)\n",
                    "zips = glob.glob('/kaggle/input/**/framework.zip', recursive=True)\n",
                    "if zips:\n",
                    "    os.makedirs(work_dir, exist_ok=True)\n",
                    "    with zipfile.ZipFile(zips[0], 'r') as zip_ref:\n",
                    "        zip_ref.extractall(work_dir)\n",
                    "    os.chdir(work_dir)\n",
                    "    print('✓ Unzipped framework from:', zips[0])\n",
                    "else:\n",
                    "    matches = glob.glob('/kaggle/input/**/run_kaggle.py', recursive=True)\n",
                    "    if matches:\n",
                    "        src_dir = os.path.dirname(matches[0])\n",
                    "        shutil.copytree(src_dir, work_dir)\n",
                    "        os.chdir(work_dir)\n",
                    "        print('✓ Framework loaded from:', src_dir)\n",
                    "    else:\n",
                    "        print('⚠️ Framework source not found in /kaggle/input/!')\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Execute pipeline command\n",
                    f"!{command_str}\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Copy all results & model checkpoints to top-level /kaggle/working/ for Kaggle output capture\n",
                    "import os, shutil\n",
                    "res_dir = '/kaggle/working/framework/results'\n",
                    "if os.path.exists(res_dir):\n",
                    "    for item in os.listdir(res_dir):\n",
                    "        s = os.path.join(res_dir, item)\n",
                    "        d = os.path.join('/kaggle/working', item)\n",
                    "        if os.path.isdir(s):\n",
                    "            shutil.copytree(s, d, dirs_exist_ok=True)\n",
                    "        else:\n",
                    "            shutil.copy2(s, d)\n",
                    "    print('✓ All results and model checkpoints successfully exported to /kaggle/working/!')\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
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
        "accelerator": "T4x2",
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
