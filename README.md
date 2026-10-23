# 🧬 NeuralOps v1.6.0

**A lightweight, correctness-gated framework for attention surgery, low-rank model compression, and hardware-accelerated kernels.**

[![PyPI version](https://badge.fury.io/py/neuralops.svg)](https://pypi.org/project/neuralops/)
[![Release](https://img.shields.io/badge/release-v1.6.0-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v1.6.0)
[![Python Versions](https://img.shields.io/pypi/pyversions/neuralops.svg)](https://pypi.org/project/neuralops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Quality](https://img.shields.io/badge/quality-gated-blue)](https://github.com/tahamajs/NeuralOps/actions)

---

## Overview

NeuralOps provides tools for optimizing and modifying Transformer architectures in PyTorch:

- **In-Place Attention Surgery**: Swap standard quadratic $O(T^2)$ self-attention layers with subquadratic, linear, recurrent, or gated alternatives with minimal calibration and fine-tuning.
- **Model Compression**: Reduce model parameter footprints and compute cost through low-rank matrix decomposition (SVD, Fisher-Weighted SVD, AdaSVD) and pruning techniques.
- **3-Level Kernel Dispatch**: Hardware-accelerated operators automatically route to Native CUDA C++ (Level 1) or Triton JIT (Level 2), with fallback to vectorized PyTorch references (Level 3) on CPU and Apple Silicon.
- **PEP 562 Lazy Loading**: Instantaneous package import (`<30ms`) with zero torch or accelerator overhead until specific layers or engines are instantiated.

### Repository Invariants

- **131 Attention Mechanisms** implemented in [`neuralops/layers/`](neuralops/layers/) across 12 mathematical families (linear, recurrent, gated, sliding window, state space, etc.).
- **72 Compression Engines** in [`neuralops/compression/`](neuralops/compression/) across 7 paradigms (SVD, Fisher, Hadamard, low-rank, pruning).
- **114 Native CUDA C++ Kernels** across 112 packages in [`neuralops/ops/cuda/`](neuralops/ops/cuda/).
- **26 Triton Kernel Modules** housing 62 unified operators across 4 tiers in [`neuralops/ops/triton/`](neuralops/ops/triton/).
- **29 Master Jupyter Notebooks** in [`notebooks/`](notebooks/) covering end-to-end workflows.

---

## Installation

### Standard / CPU Runtime
```bash
pip install neuralops
```

### NVIDIA GPU Runtime (CUDA + Triton)
```bash
pip install "neuralops[cuda]"
```

### Full Development & Verification Suite
```bash
git clone https://github.com/tahamajs/NeuralOps.git
cd NeuralOps
pip install -e ".[dev]"
```

---

## 30-Second Quickstart

### 1. In-Place Attention Surgery

Replace attention mechanisms in any Hugging Face model in-place:

```python
import torch
from transformers import AutoModelForCausalLM
from neuralops import replace_attention

# Load any Hugging Face transformer
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Swap standard quadratic attention with subquadratic cosformer linear attention
modified_model = replace_attention(
    model,
    target_attention="cosformer",
    calibration_data=None,  # Optional: pass sample inputs for closed-form ridge calibration
)
```

### 2. Model Compression via SVD Decomposition

Compress linear projection layers to a target rank:

```python
from neuralops import compress

# Apply low-rank Fisher-Weighted SVD compression across linear layers
compressed_model = compress(
    model,
    method="fwsvd",
    rank=32,
)
```

### 3. Direct Layer Instantiation

Import individual attention layers directly:

```python
import torch
from neuralops.layers.linear.cosformer import CosformerAttention

layer = CosformerAttention(dim=768, num_heads=12)
x = torch.randn(2, 64, 768)
out = layer(x)
assert out.shape == x.shape
```

---

## Software Architecture

NeuralOps is organized into four core tiers:

```
neuralops/
├── layers/              # 131 attention mechanism implementations
│   ├── linear/          # Cosformer, Linear Attention, Performers, etc.
│   ├── recurrent/       # RWKV, RetNet, GLA, HGRN, etc.
│   ├── gated/           # Gated DeltaNet, Gated Linear Attention, etc.
│   ├── sparse/          # Longformer, BigBird, Sparse Attention, etc.
│   └── hybrid/          # Multi-scale, hierarchical, and hybrid layers
├── compression/         # 72 model compression algorithms
│   ├── svd/             # Standard, Fisher-weighted, and adaptive SVD
│   ├── low_rank/        # Factorized representations
│   └── pruning/         # Magnitude and activation-aware pruning
├── ops/                 # Hardware kernel execution backends
│   ├── cuda/            # 112 native CUDA C++ kernel packages (114 implementations)
│   ├── triton/          # 26 Triton kernel modules (62 unified operators)
│   └── pytorch/         # Universal vectorized CPU/MPS fallbacks
├── trainer/             # Distillation, multi-dataset streaming, and optimizers
└── cli/                 # Command-line interface suite
```

---

## Command-Line Interface (CLI)

NeuralOps includes a CLI for inspection, diagnostics, compression, and training:

```bash
# Check environment and hardware capabilities
neuralops doctor

# Inspect canonical counts and system info
neuralops info

# List registered attention mechanisms and compression engines
neuralops list --type attentions
neuralops list --type compressors

# Run model compression from the terminal
neuralops compress --model gpt2 --method fwsvd --rank 32 --output-dir ./compressed_model
```

---

## Verification & Testing

NeuralOps includes test suites designed to run in diverse execution environments:

```bash
# Run CPU verification suite (lazy loading, architecture, distributed, code quality)
pytest tests/test_lazy_architecture.py tests/test_phase2_migration.py tests/test_code_quality.py tests/distributed/ -v

# Run codebase hygiene and invariant audit
python3 scripts/utils/audit_dead_code.py
python3 scripts/audit_readme_consistency.py
```

---

## Contributing

Please review [CONTRIBUTING.md](CONTRIBUTING.md) and [AGENTS.md](AGENTS.md) for engineering standards, commit attribution rules, and pull request checklists.

---

## License

NeuralOps is released under the [MIT License](LICENSE).
