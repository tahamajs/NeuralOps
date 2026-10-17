---
title: NeuralOps Documentation & Research Hub
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: mit
---

# 📚 NeuralOps Documentation & Research Hub

[![PyPI version](https://badge.fury.io/py/neuralops.svg)](https://pypi.org/project/neuralops/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/neuralops.svg)](https://pypi.org/project/neuralops/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neuralops.svg)](https://pypi.org/project/neuralops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/tahamajs/SVD_linear_Attention/actions/workflows/ci.yml/badge.svg)](https://github.com/tahamajs/SVD_linear_Attention/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tahamajs.github.io/SVD_linear_Attention/)

Welcome to the **NeuralOps Research Documentation Hub** (`NeuralOps v1.4.0`). This repository provides the technical specifications, architectural designs, API references, empirical benchmarks, and guides for the **SOTA gap closure suite**, formal mathematical proofs, dual 60-page conference papers, interactive 3D architecture visualizer, **131 Attention Mechanisms**, **72 Post-Training Compression Engines**, and **26 Triton Hardware Kernels**.

---

## 🧭 Core Documentation Map

| Guide | File | Purpose |
|:---|:---|:---|
| **Quick Start** | [`QUICKSTART.md`](./QUICKSTART.md) | 5-minute setup, attention injection, and compression guides |
| **Release Report (v1.4.0)** | [`reports/V1_4_0_RELEASE_REPORT.md`](./reports/V1_4_0_RELEASE_REPORT.md) | Architectural invariants, Ada Lovelace benchmarks, and camera-ready monograph |
| **Mathematical Proofs** | [`proofs/`](./proofs/) | 5 rigorous theorems: ASVD streaming, learned block-sparse DSA, FP8 E4M3 error, REAP MoE pruning, SVD-LLM v2 |
| **Conference Papers** | [`reports/NEURALOPS_MASTER_PAPER.md`](./reports/NEURALOPS_MASTER_PAPER.md) | 60-page dual-conference camera-ready publication monographs and empirical proofs |
| **3D Architecture Visualizer** | [`visualizer/`](./visualizer/) | Interactive WebGL/Three.js 3D parameter and memory manifold explorer |
| **Architecture** | [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Package structure, data flow, and registry design principles |
| **API Reference** | [`API_REFERENCE.md`](./API_REFERENCE.md) | Public API contracts for attention, compression, and inference |
| **HTML API Documentation** | [`api/neuralops/`](./api/neuralops/) | Full module-by-module auto-generated HTML API docs |
| **Library Comparison** | [`COMPARISON.md`](./COMPARISON.md) | Matrix comparing NeuralOps against FlashAttention, FLA, xformers, and vLLM |
| **Empirical Results** | [`COMPARISON_RESULTS.md`](./COMPARISON_RESULTS.md) | Comparative execution speed and latency metrics |
| **Memory Profiles** | [`MEMORY_PROFILE.md`](./MEMORY_PROFILE.md) | Peak memory allocations across attention architectures |
| **Migration Guide** | [`MIGRATION_GUIDE.md`](./MIGRATION_GUIDE.md) | Upgrading from v0.x / legacy codebases to NeuralOps v1.x |
| **Rename Map** | [`RENAME_MAP.md`](./RENAME_MAP.md) | Catalog of 657 deprecated alias mappings to canonical keys |
| **Troubleshooting** | [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md) | Solutions for CUDA OOM, Triton setup, and import conflicts |
| **Hugging Face Hub** | [`HUGGINGFACE_UPLOAD.md`](./HUGGINGFACE_UPLOAD.md) | Workflow for exporting and publishing compressed models |
| **Module Reference** | [`MODULE_REFERENCE.md`](./MODULE_REFERENCE.md) | Complete directory of attention mechanisms and compressors |

---

## 📁 Repository Directories

- **`docs/proofs/`**: Mathematical proofs validating stability, quantization error, and pruning convergence.
- **`docs/reports/`**: Monograph releases, audit roadmaps, verification logs, and camera-ready publications.
- **`docs/visualizer/`**: 3D interactive hardware and tensor flow visualizer.
- **`docs/developer_notes/`**: Diagnostic profiles, module gap analyses, and coverage notes.
- **`docs/internal_notes/`**: Internal research notes, tables, and visualization mappings.
- **`docs/api/`**: Static HTML reference site generated for GitHub Pages.
- **`docs/content/`**: Markdown guides, tutorials, and mathematical derivations.

---

## 🚀 Quick Install

```bash
# Production install (NVIDIA GPU with Triton)
pip install neuralops[cuda]

# Apple Silicon (Metal Performance Shaders)
pip install neuralops[mps]

# CPU Only
pip install neuralops[cpu]
```
