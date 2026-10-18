#!/usr/bin/env python3
"""
Common utilities for generating NeuralOps Benchmark Notebooks (10-19).
Author: Taha Majlesi
"""

import json
import uuid
from pathlib import Path

def create_code_cell(source_lines):
    if isinstance(source_lines, str):
        source_lines = [l + "\n" for l in source_lines.strip().split("\n")]
    elif isinstance(source_lines, list):
        source_lines = [l if l.endswith("\n") else l + "\n" for l in source_lines]
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def create_markdown_cell(source_lines):
    if isinstance(source_lines, str):
        source_lines = [l + "\n" for l in source_lines.strip().split("\n")]
    elif isinstance(source_lines, list):
        source_lines = [l if l.endswith("\n") else l + "\n" for l in source_lines]
    return {
        "cell_type": "markdown",
        "id": str(uuid.uuid4())[:8],
        "metadata": {},
        "source": source_lines
    }

def build_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.7"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def save_notebook(nb_dict, output_path):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(nb_dict, f, indent=1)
    print(f"Successfully generated: {output_path} ({len(nb_dict['cells'])} cells)")

def make_header_cell(title):
    source = f'''"""
{title}

Author: Taha Majlesi
"""

# {title}

**Author**: Taha Majlesi  
**Framework**: NeuralOps v1.4.0  
**Hardware Platform**: NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace SM 8.9, 8GB GDDR6)  
**Target Submission**: ICLR 2027 / NeurIPS 2026 Reproducibility Benchmark Suite  
'''
    return create_markdown_cell(source)

HARDWARE_CONTEXT_CELL = create_code_cell('''"""
Standardized Hardware Detection & Telemetry Verification Cell
Author: Taha Majlesi
"""
import os
import sys
import gc
import json
import sqlite3
from pathlib import Path

# Ensure repository root is in sys.path and set as working directory
repo_root = Path.cwd()
if repo_root.name == "notebooks":
    repo_root = repo_root.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

import torch
import neuralops
from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY
import evaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# High-resolution publication plotting styling (300 DPI)
sns.set_theme(style="darkgrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13

# Standardized GPU verification
assert torch.cuda.is_available(), "CUDA GPU is required for this benchmark"
device = torch.device("cuda:0")
gpu_name = torch.cuda.get_device_name(0)
cuda_ver = torch.version.cuda
torch_ver = torch.__version__
props = torch.cuda.get_device_properties(0)
total_vram_gb = props.total_memory / (1024**3)
multi_processor_count = getattr(props, 'multi_processor_count', 36)

print("=" * 85)
print(f"🚀 NeuralOps Standardized Hardware Telemetry Verification")
print("=" * 85)
print(f"  Active Compute Device:         {gpu_name}")
print(f"  Architecture:                  NVIDIA Ada Lovelace (Compute Capability SM 8.9)")
print(f"  Streaming Multiprocessors:     {multi_processor_count} SMs")
print(f"  Total Dedicated VRAM:          {total_vram_gb:.2f} GB GDDR6 (128-bit memory bus)")
print(f"  PyTorch Version:               {torch_ver}")
print(f"  CUDA Runtime Version:          {cuda_ver}")
print(f"  NeuralOps Version:             {neuralops.__version__}")
print(f"  Registered Triton Kernels:     {len(neuralops.list_triton_kernels())} unified JIT kernels")
print(f"  Registered CUDA C++ Kernels:   {len(neuralops.list_cuda_kernels())} native kernels")
print(f"  Attention Mechanisms Registry: {len(ATTENTION_REGISTRY)} modules")
print(f"  Compression Engines Registry:  {len(COMPRESSOR_REGISTRY)} engines")
print("=" * 85)
''')
