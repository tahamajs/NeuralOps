# 🧬 NeuralOps v2.1.0

[![PyPI version](https://img.shields.io/pypi/v/neuralops.svg)](https://pypi.org/project/neuralops/)
[![Release Badge](https://img.shields.io/badge/release-v2.1.0-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v2.1.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI Status](https://github.com/tahamajs/NeuralOps/actions/workflows/ci.yml/badge.svg)](https://github.com/tahamajs/NeuralOps/actions)

**NeuralOps** is a production-grade systems and deep learning research platform for sub-quadratic attention architectures, mathematical model compression, and hardware-fused GPU kernel optimization. Designed for ML systems engineers and researchers, NeuralOps enables 1-line surgical attention transplants into Hugging Face transformers, closed-form activation-whitened SVD compression, and high-throughput LLM serving via paged KV-cache management and self-speculative decoding.

---

## 1. Installation

Install the canonical release directly from PyPI:

```bash
pip install neuralops
```

For development or contributing from source:

```bash
pip install -e ".[dev]"
```

*Accelerated CUDA C++ extensions and Triton JIT kernels activate automatically when NVIDIA GPU hardware and driver toolchains are present.*

---

## 2. 30-Second Quickstart

Verify your installation, hardware runtime capabilities, and registered module counts:

```bash
neuralops info
```

Inspect an execution plan for an attention module without allocating large GPU tensors:

```bash
neuralops run --benchmark attention:cosformer --dry-run
```

---

## 3. Core Commands

NeuralOps exposes a unified, single-entrypoint CLI surface across benchmarking, model surgery, inference serving, and cluster diagnostics:

| Command | Description | Example |
| :--- | :--- | :--- |
| **`info`** | Inspect runtime hardware accelerator, PyTorch version, and canonical inventory | `neuralops info` |
| **`doctor`** | Audit GPU environment, compute capability, CUDA compiler, and toolchains | `neuralops doctor` |
| **`run`** | Execute isolated benchmark sweeps with automated watchdog timeouts | `neuralops run --benchmark attention:cosformer --dry-run` |
| **`benchmark`** | Dedicated benchmark group for module cataloging, planning, and runs | `neuralops benchmark list` |
| **`matrix`** | Multi-dimensional grid execution across shapes, precisions, and masking | `neuralops matrix --plan` |
| **`serve`** | Continuous batching and paged KV-cache LLM inference server | `neuralops serve --help` |
| **`compress`** | In-place SVD, Hadamard Incoherent SVD, and low-rank weight factorization | `neuralops compress --help` |

---

## 4. Architectural Catalog & Documentation

NeuralOps maintains mathematically validated registries across every supported operator:
- **Attention Mechanisms (131 modules):** Linear attention, state-space models (SSM), gated recurrences, and hardware-fused kernels cataloged in [`docs/api/attentions.md`](docs/api/attentions.md).
- **Compression Engines (72 engines):** Truncated SVD, Fisher-weighted spectral pruning, and activation whitening documented in [`docs/api/compression.md`](docs/api/compression.md).
- **GPU Kernels (166 units):** 140 CUDA C++ and 26 canonical Triton JIT fusion operators detailed in [`docs/api/kernels.md`](docs/api/kernels.md).

For deep architectural analyses, empirical benchmark ledgers, and comprehensive developer documentation:
- 📖 [Full Technical & Architectural Reference Manual](docs/README_DETAIL.md)
- 📊 [Master Empirical Results Ledger](docs/api/results_ledger.md)
- 🔬 [Interactive Demonstration Tutorials & Examples](examples/README.md)
- 📝 [Release History & Migration Guide](CHANGELOG.md)
- 🛡️ [Security Policy & Advisory Disclosure](SECURITY.md)
- 🤝 [Contributing Guidelines & Quality Gates](CONTRIBUTING.md)

---

## 5. Citation

If you incorporate NeuralOps architectures, SVD compression pipelines, or fused kernels into your research, please cite:

```bibtex
@software{Majlesi_NeuralOps_2026,
  author = {Majlesi, Taha},
  title = {{NeuralOps: High-Performance Attention Zoo, SVD Compression, and Fused Triton Kernels}},
  year = {2026},
  url = {https://github.com/tahamajs/NeuralOps},
  version = {2.1.0}
}
```

---

## 6. License

This repository is licensed under the [MIT License](LICENSE).
