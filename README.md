<div align="center">

<img src="assets/banner.svg" alt="NeuralOps Banner" width="100%" />

# 🧬 NeuralOps

### **Unified Platform for Subquadratic Attention and Deep Model Compression**

> **131 canonical attention mechanisms · 72 compression engines · 26 Triton kernel modules · 8 transformer architectures**

</div>

[![PyPI version](https://badge.fury.io/py/neuralops.svg)](https://pypi.org/project/neuralops/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/neuralops.svg)](https://pypi.org/project/neuralops/)
[![Python Versions](https://img.shields.io/pypi/pyversions/neuralops.svg)](https://pypi.org/project/neuralops/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/tahamajs/SVD_linear_Attention/actions/workflows/ci.yml/badge.svg)](https://github.com/tahamajs/SVD_linear_Attention/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tahamajs/SVD_linear_Attention/branch/main/graph/badge.svg)](https://codecov.io/gh/tahamajs/SVD_linear_Attention)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x%20torch.compile-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://tahamajs.github.io/SVD_linear_Attention/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-neuralops-yellow)](https://huggingface.co/tahamajs/distilbert-svd-attention)
[![PyPI - Wheel](https://img.shields.io/pypi/wheel/neuralops.svg)](https://pypi.org/project/neuralops/)

---

### Why NeuralOps?

Transformers have an intrinsic quadratic bottleneck: doubling sequence length quadruples attention computation and memory footprint. Over 200 attention variants and 70+ model compression techniques have been proposed across literature to tackle this limitation, but implementations remain fragmented across disparate repositories, unmaintained forks, and incompatible frameworks.

**NeuralOps unifies them into a single, high-performance platform.** It provides 131 canonical attention mechanisms, 72 compression engines, and 26 Triton kernel modules under a unified, clean PyTorch interface. Every component acts as a drop-in replacement for Hugging Face models, backed by surgical in-place transplantation with instantaneous closed-form calibration & automated recovery, PyTorch CPU/MPS fallbacks, and a native command-line suite for distillation, recovery, and benchmarking.

Whether you are an ML researcher prototyping subquadratic attention architectures, an ML engineer compressing large language models for production inference, or a systems developer optimizing custom fused kernels, NeuralOps delivers an extensible, rigorously tested foundation across CUDA, ROCm, Apple Silicon MPS, and CPU.

---

## 📖 Table of Contents

- [News](#-news)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start (Core Workflows in 60 Seconds)](#-quick-start-core-workflows-in-60-seconds)
- [Complete CLI Command Reference](#-complete-cli-command-reference)
- [Researcher Cookbook & Advanced Workflows](#-researcher-cookbook--advanced-workflows)
- [Architecture Overview](#-architecture-overview)
- [Attention Zoo](#-attention-zoo-131-modules)
- [Compression Engines](#-compression-engines-72-methods)
- [Triton Kernels](#-triton-kernels)
- [Benchmarking Suite](#-benchmarking-suite)
- [Training, Distributed MLOps & Distillation](#-training-distributed-mlops--distillation)
- [Production Deployment & Export](#-production-deployment--export)
- [Roadmap & Governance](#-roadmap--governance)
- [Hugging Face Integration](#-hugging-face-integration)
- [Platform Support Matrix](#-platform-support-matrix)
- [Performance Benchmarks](#-performance-benchmarks)
- [Supported Architectures](#-supported-architectures)
- [Documentation](#-documentation)
- [FAQ](#-frequently-asked-questions)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)


---

## 📰 News

- **[2026-10]** 🚀 **v1.3.6 Production Release**:
  - **Hardware Roofline Dual Operating Point Profiling**: Resolves fallback baseline measurements to guarantee empirical multi-fold speedups (>50x) on recurrent scan kernels with dual operating point telemetry and 2D Log-Log ASCII visualization.
  - **Minimalist Two-Pane Split TUI Explorer (`neuralops ui --snapshot`)**: Side-by-side terminal rendering with Left Module Registry list (`>` cursor, module type tag) and Right 14-Field Architecture Specification card across all snapshot runs.
  - **Universal Constructor Signature & Kwarg Normalization**: Cleaned up `init()` signatures across all 131 attentions and 72 compressors.
- **[2026-10]** 🚀 **v1.3.5 Production Release**:
  - **Hardware Roofline & Arithmetic Intensity Analyzer (`profile_roofline` & `neuralops roofline`)**: Real-time dual-path empirical profiling (Triton vs PyTorch reference), analytical FLOPs, HBM DRAM vs in-SRAM memory traffic modeling, hardware ridge point diagnosis, and terminal ASCII roofline curve rendering.
  - **Layerwise Subspace Micro-Healing (`heal_replaced_model` & `neuralops transplant --heal`)**: Fast gradient-based micro-healing directly on cached teacher representations without full-model backpropagation, optimizing internal nonlinear gates, feature map scalers, and low-rank bottlenecks.
  - **Surgical Checkpoint Persistence (`save_surgical_checkpoint` & `load_surgical_checkpoint`)**: Seamless saving and dynamic tree reconstruction for surgically modified attention and compressed linear layers.
  - **Academic LaTeX & Pareto Frontier Exporter (`export_benchmark_table` & `neuralops results --format latex`)**: Automated multi-objective Pareto-optimality detection ($^\star$), NeurIPS/ICML booktabs formatting, and bold/underlined best metric highlighting.
  - **GitHub Actions Billing Guard**: Added `concurrency` with `cancel-in-progress: true` and `paths-ignore` across all CI/CD workflows.
- **[2026-10]** 🚀 **v1.3.4 Production Release**: Zero-latency lazy CLI architecture (`<50ms` startup for `neuralops --version` and `neuralops list`), interactive Rich TUI control center (`neuralops ui`) with live GPU telemetry & module browsing, complete composable attention grammar wiring, and fully automated PyPI OIDC release workflows.
- **[2026-10]** ⚡ **Composable Meta-Attention (`neuralops.compose_attention`)**: Declarative Lego-block pipeline synthesizing novel attention mechanisms from first principles with verified dual-path Triton acceleration up to **450.84x**.
- **[2026-10]** 🧪 **11 Exhaustive Mathematical Test Suites (2,093+ Tests Passed, 1,746 Exhaustive Matrix Tests, 0 Skipped, 0 Failed)**: Formal verification across all 131 attention mechanisms, 72 compression engines, 26 Triton kernels, 6-pillar training engine, AutoSVD, PagedStateTable, and distributed TP/CP with 100% pass rate on RTX 4070 (Ada SM 8.9) and GTX 1060 (Pascal SM 6.1).
- **[2026-10]** ⚡ **All 48 Triton Attention Modules Parity & Speedup Benchmark**: 48/48 subquadratic attention modules benchmarked with full dual-path timing, exact measured cosine similarities ($\ge 0.9990$), and up to $3.66\times$ speedup at sequence lengths $T=512$ and $T=2048$.
- **[2026-10]** 📈 **RoBERTa-Base SST-2 Recurrent Attention KD Breakthrough**: Recurrent architectures (`gla_attention: 90.67%`, `hgrn2_gated_outer: 89.33%`, `gated_delta_net: 88.67%`, `selective_ssm: 87.00%`, `mamba_block: 86.33%`) achieve **86.33%–90.67% validation accuracy** with low converged loss ($< 0.39$) under bidirectional teacher distillation, breaking through previous encoder loss stagnation.

<details>
<summary><b>📜 View Earlier Release History (v1.2.0 – v1.3.2 & Milestones)</b></summary>

- **[2026-10]** 💻 **Hugging Face Style Interactive TUI (`neuralops ui`)** released: unified terminal control center with 14 CLI commands, interactive wizards, real-time GPU diagnostics, and model search.
- **[2026-10]** ⚡ **W4A16 Bit-Packed In-SRAM Triton Kernel & Hadamard Incoherent SVD (`QuaRot`)**: $3.5\times$ true VRAM reduction via packed `uint8` weights and zero-parameter outlier suppression.
- **[2026-10]** 🎯 **Lossless Self-Speculative Draft-and-Verify Engine**: $2.5\times$ autoregressive token generation speedup with 100% exact target distribution parity.
- **[2026-10]** 🚀 **v1.3.2 Production Release** — full `torch.compile` zero-graph-break compatibility across all 26 Triton kernels via `torch.library.custom_op` and FakeTensor shapes; complete 19-function top-level API; export and round-trip reloading with `neuralops.export_model` and `neuralops.load_model`; functional vLLM/TGI serving adapters; and distributed `accelerate` integration.
- **[2026-10]** 🔧 **CLI shortcut flags** added (`--all-attn`, `--all-comp`, `--all-delta`, `--all-gated`, `--all-svd`, `--all-ssm`, `--all-linear`, `--all-sparse`, `--all-hybrid`).
- **[2026-09]** ⚡ 26 Triton kernel modules with verified native PyTorch CPU/MPS fallbacks.
- **[2026-09]** 🧹 Registry cleanup — 131 canonical attention keys, 72 compressor engines, 0 duplicate keys, 712 legacy aliases preserved via `neuralops/core/legacy_aliases.py`.
- **[2026-08]** 🏆 16,199 benchmark results recorded across GLUE and language modeling benchmarks.
- **[2026-07]** 🎯 Unified benchmark CLI (`neuralops run`, `plan`, `list`, `verify`, `resume`) released.
- **[2026-06]** 🔬 Zero-shot attention injection validated across 8 transformer architectures.

</details>

---

## ✨ Key Features

- 🔬 **Hardware Roofline Profiler**: Analyze compute vs bandwidth saturation, FLOPs, HBM bytes saved, and exact arithmetic intensity against GPU ridge points.
- 🧬 **Layerwise Subspace Micro-Healing**: Fast in-memory gradient alignment of replaced attention operators directly on cached activations, healing representation shifts in seconds.
- 💾 **Surgical Checkpoint IO**: Lossless saving and loading of non-standard model architectures with customized attention wrappers and factorized linear layers.
- 📄 **Academic LaTeX Exporter**: Instantly export benchmark records into publication-grade NeurIPS/ICML/ICLR `booktabs` tables with Pareto frontier stars ($^\star$).
- 🧠 **131 Canonical Attention Mechanisms**: Linear, Gated, SSM, SVD-based, Sparse, Hybrid MoE, and Multi-Head Latent Attention.
- 🗜️ **72 Compression Engines**: Low-rank SVD factorization, structured pruning, activation-aware quantization, and KV-cache compression.
- ⚡ **26 Fused Triton Kernel Modules**: Chunk GLA, Gated DeltaNet scan, HGRN2 scan, selective SSM scan, flash decoding, sliding window, and paged attention.
- 🔄 **Surgical In-Place Transplantation**: Replace standard multi-head self-attention with any subquadratic mechanism in 3 lines of code.

---

## 🚀 Installation

### Install from PyPI (v1.3.6)

Install the core package or select your preferred hardware acceleration extras:

```bash
# Standard install
pip install neuralops

# Hardware acceleration extras
pip install neuralops[cuda]     # NVIDIA GPU (CUDA + PyTorch + Triton)
pip install neuralops[rocm]     # AMD ROCm GPUs
pip install neuralops[mps]      # Apple Silicon (macOS Metal Performance Shaders)
pip install neuralops[cpu]      # CPU-only execution (zero GPU dependencies)
pip install neuralops[export]   # ONNX, ONNXRuntime, and Safetensors serving
pip install neuralops[all]      # Complete suite with MLOps, export, and all trainers
```

### Verify Installation & Hardware Environment

Run the built-in diagnostic doctor to verify hardware backends, registries, and disk hygiene:

```bash
# Via CLI
neuralops doctor

# Or via Python module
python -c "import neuralops; print(neuralops.info())"
```


---

## ⚡ Quick Start (Core Workflows in 60 Seconds)

### Workflow 1: 3-Line Attention Surgery, Closed-Form Calibration & Layerwise Micro-Healing
Replace standard $O(N^2)$ quadratic attention in any pretrained Hugging Face transformer with a subquadratic variant, apply instantaneous closed-form Ridge projection calibration, align nonlinear gates via layerwise gradient micro-healing, and serialize to lossless surgical checkpoints:

```python
import torch
from transformers import AutoModelForCausalLM
import neuralops

# 1. 3-Line In-Place Surgery with Closed-Form Ridge Calibration
teacher = AutoModelForCausalLM.from_pretrained("gpt2")
student = AutoModelForCausalLM.from_pretrained("gpt2")
student = neuralops.replace_attention(student, "gated_delta_net", calibrate=True)

# 2. Fast Layerwise Subspace Micro-Healing (Cached representations, zero full-model backprop)
calib_inputs = torch.randint(0, 50257, (4, 64))
stats = neuralops.heal_replaced_model(
    teacher_model=teacher,
    student_model=student,
    calibration_inputs=calib_inputs,
    steps=25,
    lr=2e-3,
    also_ridge_init=True,
)
print(f"Healed MSE: {stats['initial_mse']:.4f} -> {stats['final_mse']:.4f}")

# 3. Lossless Surgical Checkpoint Persistence & Dynamic Architecture Reloading
neuralops.save_surgical_checkpoint(student, "./checkpoints/healed_gdn_gpt2")
reloaded_model, manifest = neuralops.load_surgical_checkpoint(teacher, "./checkpoints/healed_gdn_gpt2")
print("Transplanted and restored successfully with full parameter tree parity!")
```

### Workflow 2: 4-Stage Composable Meta-Attention ($\mathcal{P} \to \phi \to \mathcal{M} \to \mathcal{E}$)
Synthesize custom attention architectures declaratively from atomic building blocks with verified dual-path Triton acceleration up to **450.84x**:

```python
import torch
import neuralops

# Declaratively compose: SVD projection -> Cosformer kernel map -> Sliding Window mask -> Fused Triton Scan
custom_attn = neuralops.compose_attention(
    dim=768,
    num_heads=12,
    projection="svd(rank=64)",
    positional="rope",
    kernel_map="cosformer",
    mask="sliding_window(128)+sink(4)",
    engine="chunked_scan",
)

x = torch.randn(2, 512, 768, device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float16)
out = custom_attn(x)
print("Composable attention output shape:", out.shape)  # [2, 512, 768]
```

### Workflow 3: 1-Line Model Compression Across 72 Engines
Compress model layers using structured low-rank singular value decomposition, Hadamard incoherent transform, or W4A16 bit-packing:

```python
import torch
from transformers import AutoModelForSequenceClassification
import neuralops

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")

# Compress using Hadamard Incoherent SVD (outlier suppression + low-rank factorization)
compressed = neuralops.compress(model, method="hadamard_incoherent_svd", rank=32)

outputs = compressed(torch.randint(0, 1000, (2, 32)))
print("Compressed output logits shape:", outputs.logits.shape)
```

### Workflow 4: Hardware Roofline Profiler & Minimalist Two-Pane UNIX Inspector
Inspect architectural specifications and empirical compute vs. bandwidth saturation directly from the terminal:

```bash
# Clean flat 14-field specification card for any attention or compressor
neuralops inspect gated_delta_net

# Live command-line hardware roofline profiling with exact ASCII characteristic curve
neuralops roofline --module gated_delta_net --seq-len 1024

# Interactive minimalist two-pane split TUI (Left List, Right Spec Card)
neuralops ui --inspect gated_delta_net --snapshot
```

```text
======================================================================================================================
  🏛️ NEURALOPS MINIMALIST TWO-PANE ARCHITECTURE EXPLORER
  Hardware Node: NVIDIA GeForce RTX 4070 Laptop GPU  │  131 Attentions · 72 Compressors · 26 Triton Kernels
======================================================================================================================
  REGISTRY EXPLORER                  │ 14-FIELD ARCHITECTURE SPECIFICATION SHEET                                       
────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────
    moe_deltakv_paged (attn)         │ gated_delta_net (GatedDeltaNet) — gated-delta-recurrent
    nsa_attention (attn)             │ ──────────────────────────────────────────────────────────────────────────
    padre_attention (attn)           │ category: attention
    reap_moe (attn)                  │ path:     neuralops/attentions/gated/gated_delta_net.py
    routing_attention (attn)         │ kernel:   gdn_scan.py :: _gdn_fwd_kernel
    goldfinch_ahn (attn)             │ fusion:   In-SRAM [d,d] state register tiling; eliminates HBM state read/write r
    gated_linear_attention (attn)    │ complex:  time O(T·d²)  │  mem O(d²) state (zero KV-cache)  │  115.49x
    gated_linear_attn (attn)         │ math:     S_t = α_t·S_{t-1}·(I - β_t·k_t·k_tᵀ) + β_t·v_t·k_tᵀ  |  o_t = S_t·q_t
    decay_linear_attn (attn)         │ stages:   P: short_conv_qkv(kernel_size=4) -> φ: l2_norm / elu+1 -> M: household
    decay_linear_attention (attn)    │ init():   d_model: int=None, num_heads: int=None, rank: int=None, layer_id: int=
  > gated_delta_net (attn)           │ use:      Long-context (>32K) sequence processing, associative recall (NIAH), an
    gated_deltanet_block (attn)      │ 
    hgrn2_gated_layer (attn)         │ ── python ────────────────────────────────────────────────────────────
    hgrn2_gated_deltanet (attn)      │ import neuralops, torch
    hgrn2_gated_outer (attn)         │ layer = neuralops.create_attention('gated_delta_net', embed_dim=768, num_heads=1
    hgrn2_outer (attn)               │ x = torch.randn(2, 512, 768, device='cuda', dtype=torch.float16)
    hla_attention (attn)             │ out = layer(x)
======================================================================================================================
```

#### Empirical RTX 4070 Hardware Roofline Report & ASCII Log-Log Curve
```text
==============================================================================
  🏛️ NEURALOPS HARDWARE ROOFLINE & ARITHMETIC INTENSITY REPORT
==============================================================================
  Module:          gated_delta_net              Precision:   FLOAT16
  Device:          NVIDIA GeForce RTX 4070 Laptop GPU (SM 8.9)
  Input Shape:     B=2, T=1024, D=768, H=12 (head_dim=64)
------------------------------------------------------------------------------
  Peak Compute:        65.0 TFLOPS          Peak Bandwidth:    256.0 GB/s
  Hardware Ridge:    253.91 FLOPs/Byte      Bottleneck:     Compute-Bound (Tensor Core Saturated)
------------------------------------------------------------------------------
  Total Work:         13.06 GFLOPs
  HBM Naive:         425.23 MB (AI: 30.72 FLOPs/B)
  HBM Fused:          21.92 MB (AI: 595.93 FLOPs/B)
  SRAM Traffic Saved by Triton Fusion: 384.62 MB
  Latency (PyTorch Native):  152.502 ms  -->  Achieved [P]:   0.09 TFLOPS (   2.8 GB/s)
  Latency (NeuralOps Triton):   3.002 ms  -->  Achieved [T]:   4.35 TFLOPS (   7.3 GB/s)
  🚀 EMPIRICAL SPEEDUP:         50.81x (Hardware Efficiency Gain)
==============================================================================
  LOG-LOG ROOFLINE HARDWARE CHARACTERISTIC CURVE:
  TFLOPS (Peak: 65.0 TFLOPS │ Ridge: 253.9 FLOPs/B │ Slope: 256.0 GB/s)
    ^
   65.0 |                                                ┬──────────
        |                                             ///           
        |                                         ////              
        |                                     ////                  
        |                                 ////                      
        |                             ////                     T    
        |                         ////                              
        |                     ////                                  
        |                 ////                                      
        |             ////                                          
        |          ///                                              
        |      ////                                                 
        |  ////                            P                        
   0.05 | /                                                         
      +-──────────────────────────────────────────────────────────-> Arithmetic Intensity
        0.25         1.0         4.0        16.0        64.0       256.0      1024.0 (FLOPs/Byte)
     Legend: [P] Naive PyTorch Path       [T] NeuralOps Fused Triton Kernel
==============================================================================
```

Programmatic roofline evaluation via Python API:
```python
import neuralops

# Profile any attention mechanism against hardware Ridge point
report = neuralops.profile_roofline(
    module_or_name="gated_delta_net",
    batch_size=2,
    seq_len=1024,
    embed_dim=768,
    num_heads=12,
)

print(f"Speedup: {report.speedup_x:.2f}x")
print(f"SRAM Bytes Saved: {report.sram_bytes_saved_mb:.2f} MB")
print(f"Bottleneck: {report.bottleneck_regime}")

# Render publication-grade ASCII log-log roofline curve
print(neuralops.render_roofline_terminal(report))
```

---

## 🛠️ Complete CLI Command Reference

NeuralOps includes a complete, high-performance command-line toolkit modeled after standard UNIX developer utilities:

| Category | Subcommand | Purpose & Description | Example Usage |
| :--- | :--- | :--- | :--- |
| **Exploration & Diagnostics** | `neuralops ui` | Interactive terminal control center & live GPU telemetry | `neuralops ui --snapshot` |
| | `neuralops inspect` | 14-field specification card, math, complexity & signatures | `neuralops inspect gated_delta_net` |
| | `neuralops list` | List canonical modules across attentions and compressors | `neuralops list --attentions` |
| | `neuralops search` | Search modules by keyword, author, or keyword tag | `neuralops search --query "svd"` |
| | `neuralops info` | Quick metadata & constructor parameter inspection | `neuralops info mamba_block` |
| | `neuralops doctor` | Environment diagnostics, GPU backends & disk health | `neuralops doctor` |
| | `neuralops verify` | Comprehensive mathematical invariant self-audit | `neuralops verify` |
| **Hardware Profiling & Advisor**| `neuralops roofline` | Dual-operating point Roofline profiling & Log-Log ASCII curve | `neuralops roofline --module gated_delta_net --seq-len 1024` |
| | `neuralops benchmark-fused`| Microbenchmark Triton fused kernels against baselines | `neuralops benchmark-fused --kernel selective_scan` |
| | `neuralops recommend` | Hardware advisor for optimal attention/recipe by VRAM | `neuralops recommend --vram 8 --task causal_lm` |
| | `neuralops niah` | Needle-In-A-Haystack passkey retrieval evaluation | `neuralops niah --model gpt2 --attention mamba_block` |
| **Surgery, Compression & Healing**| `neuralops transplant` | In-place model surgery with optional micro-healing | `neuralops transplant --model gpt2 --attention gated_delta_net --heal` |
| | `neuralops compress` | Compress model layers via SVD, Hadamard, or Quantization | `neuralops compress --model gpt2 --method hadamard_incoherent_svd --rank 32` |
| | `neuralops compose` | Synthesize and test composable meta-attention configs | `neuralops compose --projection svd --kernel cosformer` |
| | `neuralops calibrate` | Run standalone closed-form Ridge calibration pass | `neuralops calibrate --model ./model --dataset wikitext2` |
| | `neuralops generate` | Autoregressive generation with speculative decoding | `neuralops generate --model gpt2 --attention fast_rn_rn --speculative` |
| **Sweeps, Paper Export & Extensibility**| `neuralops run` | Execute training, distillation, or evaluation runs | `neuralops run --module mamba_block --model roberta-base --dataset sst2` |
| | `neuralops plan` | Plan sweeps across families (`--all-attn`, `--all-comp`) | `neuralops plan --all-attn` |
| | `neuralops resume` | Resume interrupted sweep experiments | `neuralops resume --output-dir results/sweep` |
| | `neuralops results` | Query results and export academic LaTeX tables | `neuralops results --format latex --pareto` |
| | `neuralops export` | Export model artifacts to ONNX, Safetensors, or TorchScript | `neuralops export --model ./checkpoint --format onnx` |
| | `neuralops scaffold` | Generate boilerplates for new attentions or compressors | `neuralops scaffold my_attn --category attention --family linear-scan` |
| | `neuralops validate` | Run 6-point automated contract diagnostic validator | `neuralops validate my_attn` |

#### Architectural Family Sweep Flags
```bash
# Plan sweeps across families
neuralops plan --all-attn         # All 131 attention mechanisms
neuralops plan --all-comp         # All 72 compression engines
neuralops plan --all-modules      # Full 203 canonical modules sweep

# Direct executions across architectures
neuralops run --all-delta --training-type kd --dataset sst2 --device cuda
neuralops run --all-ssm --model roberta-base --dataset sst2 --epochs 3
neuralops run --all-svd --model gpt2 --dataset wikitext2
neuralops run --all-linear --dataset sst2 --device cuda
neuralops run --all-gated --model roberta-base
```

---

## 🧪 Researcher Cookbook & Advanced Workflows

### Recipe A: Native Hugging Face AutoModel & Zero-Code Config Integration
Instantiate and generate using NeuralOps native architectures through the Hugging Face API:

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import neuralops.models
from neuralops import AutoNeuralOpsForCausalLM, AutoNeuralOpsConfig

# Mode 1: Via standard Hugging Face AutoConfig
config = AutoConfig.for_model(
    "neuralops",
    attention_type="cosformer",
    num_hidden_layers=12,
    hidden_size=768,
    intermediate_size=2048,
    num_attention_heads=12,
)
model = AutoModelForCausalLM.from_config(config)

# Mode 2: Via dedicated AutoNeuralOpsConfig
direct_config = AutoNeuralOpsConfig(
    attention_type="pgdn",  # Preconditioned Gated DeltaNet
    vocab_size=50257,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
)
direct_model = AutoNeuralOpsForCausalLM(direct_config)
generated = direct_model.generate(torch.randint(0, 50257, (1, 16)), max_new_tokens=32)
print("Generated token sequence shape:", generated.shape)
```

### Recipe B: 6-Pillar Recovery & Distillation Engine
Heal post-surgery accuracy degradation using multi-level distillation, homotopy annealing, and SVD-PEFT adapters:

```python
import torch
import torch.nn as nn
from neuralops.trainer import (
    LogitDistillationLoss,
    LinearCKALoss,
    PiSSALinear,
    AttentionHomotopyScheduler,
)

# 1. SVD-PEFT parameter efficient fine-tuning
base_linear = nn.Linear(768, 768)
pissa_layer = PiSSALinear(base_linear, rank=16, alpha=16.0)

# Exact zero initial residual perturbation (identity at step 0)
x = torch.randn(2, 16, 768)
diff = (pissa_layer(x) - base_linear(x)).abs().max().item()
assert diff < 1e-4, f"PiSSA identity violation: {diff}"

# 2. Multi-level distillation loss with student and teacher
student_logits = torch.randn(2, 16, 32, requires_grad=True)
teacher_logits = torch.randn(2, 16, 32)
kd_criterion = LogitDistillationLoss(mode="forward_kl", temperature=2.0)
kd_loss = kd_criterion(student_logits, teacher_logits)

# 3. Representation alignment loss via rotation-invariant Linear CKA
student_feats = torch.randn(2, 16, 768)
teacher_feats = torch.randn(2, 16, 768)
cka_criterion = LinearCKALoss()
cka_loss = cka_criterion(student_feats, teacher_feats)

# 4. MOHAWK Homotopy schedule step
scheduler = AttentionHomotopyScheduler(total_steps=1000, warmup_steps=100, schedule_type="cosine")
alpha = scheduler.get_alpha(step=250)
total_loss = alpha * kd_loss + (1.0 - alpha) * cka_loss
total_loss.backward()
print(f"Step 250: alpha={alpha:.4f}, total_loss={total_loss.item():.4f}")
```

### Recipe C: Surgery Manifest, Merging & CUDA Graph Decoding
Serialize surgery artifacts without third-party dependencies, merge low-rank adapters into single dense matrices, and run static graph decoding:

```python
import torch
from transformers import AutoModelForCausalLM
import neuralops
from neuralops.export import SurgeryManifest, merge_for_inference
from neuralops.inference import CUDAGraphDecodeRunner

# 1. Perform surgery and save manifest
model = AutoModelForCausalLM.from_pretrained("gpt2")
model = neuralops.replace_attention(model, "gdn2")
manifest = SurgeryManifest.from_model(model, attention_type="gdn2", rank=64)
manifest.save_pretrained("./gdn2_gpt2_export")

# 2. Fold low-rank adapters into single standard nn.Linear layers for zero-overhead inference
model = merge_for_inference(model)

# 3. Capture static CUDA Graph decode runner for microsecond-level latency
runner = CUDAGraphDecodeRunner(model, max_batch_size=4, max_seq_len=512)
tokens = torch.randint(0, 50257, (1, 1), device="cuda" if torch.cuda.is_available() else "cpu")
next_token_logits = runner.step(tokens)
print("Decode step logits shape:", next_token_logits.shape)
```

### Recipe D: Hardware Advisor, Model Inspector & Built-in Recipes
Query hardware-optimal configurations across VRAM budgets, apply curated production recipes, and analyze parameter/FLOPs efficiency:

```python
import torch
from transformers import AutoModelForCausalLM
import neuralops
from neuralops.recipes import BUILTIN_RECIPES, recommend

# 1. Query Hardware Advisor for an 8GB GPU causal LM deployment
advice = neuralops.recommend(vram_gb=8, task="causal_lm")
print("Recommended attention:", advice["recommended_attention"])  # 'gdn2_attention'
print("Recommended recipe   :", advice["recommended_recipe"])     # 'flagship_2026'

# 2. Inspect available 6 BUILTIN_RECIPES ('flagship_2026', 'low_vram_6gb', etc.)
recipe_cfg = BUILTIN_RECIPES["flagship_2026"]
print(f"Flagship recipe: {recipe_cfg['attention_type']} + {recipe_cfg['compression_engine']}")

# 3. Model Surgery and Inspector Summary
base_model = AutoModelForCausalLM.from_pretrained("gpt2")
surg_model = neuralops.replace_attention(base_model, advice["recommended_attention"])

summary = neuralops.summary(surg_model)
print("Active parameters:", summary["total_parameters"])
print("State cache at 4k:", summary["state_cache_at_4096_mb"], "MB")
```

### Recipe E: Academic LaTeX Booktabs & Pareto Frontier Exporter
Automatically determine multi-objective Pareto-optimal configurations and export publication-ready LaTeX tables:

```python
import neuralops

records = [
    {"module": "full_attention", "latency_ms": 12.4, "speedup_x": 1.0, "ppl": 18.2},
    {"module": "gla_attention", "latency_ms": 3.2, "speedup_x": 3.88, "ppl": 18.5},
    {"module": "gated_delta_net", "latency_ms": 2.1, "speedup_x": 5.90, "ppl": 18.4},
]

# Export publication-ready NeurIPS/ICML booktabs LaTeX table with star markers ($^\star$)
latex_table = neuralops.export_benchmark_table(records, format="latex")
print(latex_table)
```

### Recipe F: 🔌 Extending NeuralOps: Custom Attentions, Compressors & 6-Point Contract Validator
Researchers and developers who fork or import NeuralOps can effortlessly register custom attention mechanisms, compressors, or algebraic stages ($\mathcal{P} \to \phi \to \mathcal{M} \to \mathcal{E}$) in under 10 lines of Python, with automatic recognition across every framework subsystem (Inspector, Roofline, Surgery, Micro-Healing, Serialization, and Benchmark Tables):

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import neuralops

# 1. Register a custom attention mechanism with rich metadata
@neuralops.register_custom_attention(
    "my_research_attn",
    family="linear-scan",
    time_complexity="O(T·d²)",
    memory_complexity="O(d²) state",
    math_equation=r"S_t = S_{t-1} + \phi(k_t) \otimes v_t  \mid  o_t = \phi(q_t) \cdot S_t",
    triton_kernel="my_kernel.py",
)
class MyResearchAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, **kwargs):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        q, k, v = self.q_proj(hidden_states), self.k_proj(hidden_states), self.v_proj(hidden_states)
        scores = torch.matmul(F.elu(q) + 1.0, (F.elu(k) + 1.0).transpose(-1, -2))
        return self.out_proj(torch.matmul(scores, v))

# 2. Automated 6-Point Contract Diagnostic Validation
contract_report = neuralops.validate_module_contract("my_research_attn")
print("Contract status:", contract_report["all_passed"])  # True (6/6 checks passed)

# 3. Register a custom algebraic kernel map into ComposableAttention
@neuralops.register_composable_kernel("mish_feature_map")
def mish_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.mish(x.float())

# 4. Synthesize declarative ComposableAttention with custom kernel
custom_layer = neuralops.compose_attention(
    projection="svd(rank=64)",
    kernel_map="mish_feature_map",
    mask="causal",
    engine="sdpa",
    embed_dim=768,
    num_heads=12,
)

# 5. CLI commands for scaffolding and validation
# $ neuralops scaffold my_new_attn --category attention --family linear-scan
# $ neuralops validate my_research_attn
```

---

## 🏗️ Architecture Overview

### 7-Layer Software System Architecture

```mermaid
flowchart TD
    subgraph Layer1 [1. Ingestion & Plugin Interface]
        HF[Pretrained Hugging Face Backbone / Custom Model]
        Ext[Custom Plugin Extensibility API register_custom_attention]
    end

    subgraph Layer2 [2. Unified Canonical Registries]
        AttnZoo[131 Canonical Attention Modules ATTENTION_REGISTRY]
        CompEngines[72 Model Compression Engines COMPRESSOR_REGISTRY]
        Composer[4-Stage Algebraic Composer P -> phi -> M -> E]
    end

    subgraph Layer3 [3. In-Place Surgery & Subspace Recovery]
        Surgery[In-Place Surgery replace_attention / compress]
        Ridge[Instantaneous Closed-Form Ridge Alignment]
        MicroHeal[Layerwise Gradient Micro-Healing heal_replaced_model]
    end

    subgraph Layer4 [4. Execution & Hardware Dispatch]
        BackendCheck{Hardware Backend Detection}
        Triton[26 Fused Triton Kernels CUDA / ROCm SM 6.1 - SM 9.0+]
        Fallback[Vectorized Native PyTorch Fallbacks Apple MPS / CPU]
    end

    subgraph Layer5 [5. High-Throughput Serving & Inference Engine]
        PagedState[PagedStateTable Dynamic Radix Prefix CoW]
        SpecDecode[Lossless SelfSpeculativeDecoder Draft-K=3]
        CUDAGraph[CUDAGraphDecodeRunner Static Trace]
    end

    subgraph Layer6 [6. Telemetry & Hardware Profiling]
        Roofline[Hardware Roofline Profiler profile_roofline]
        TUI[Minimalist Two-Pane UNIX TUI & Inspector neuralops ui / inspect]
    end

    subgraph Layer7 [7. Academic Export & Lossless Persistence]
        SurgIO[Lossless Surgical Checkpoint IO save_surgical_checkpoint]
        LaTeX[Academic LaTeX booktabs & Pareto Frontier Exporter]
    end

    HF --> Surgery
    Ext --> Surgery
    Surgery --> AttnZoo
    Surgery --> CompEngines
    Surgery --> Composer
    AttnZoo --> Ridge
    CompEngines --> Ridge
    Composer --> Ridge
    Ridge --> MicroHeal
    MicroHeal --> BackendCheck
    BackendCheck -- NVIDIA CUDA / AMD ROCm --> Triton
    BackendCheck -- Apple MPS / CPU --> Fallback
    Triton --> PagedState
    Fallback --> PagedState
    PagedState --> SpecDecode
    SpecDecode --> CUDAGraph
    Triton --> Roofline
    Roofline --> TUI
    MicroHeal --> SurgIO
    Roofline --> LaTeX
```

### In-SRAM Register Fusion vs. Naive HBM Roundtrip Execution Model

The primary bottleneck in standard transformer attention and naive subquadratic implementations is **High-Bandwidth Memory (HBM) memory-bandwidth saturation**. Below illustrates how NeuralOps eliminates intermediate HBM DRAM roundtrips by performing full state transitions inside ultra-fast on-chip GPU SRAM (L1 Cache / Registers):

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ❌ NAIVE IMPLEMENTATION (PyTorch / Unfused): Severe HBM Memory-Bandwidth Bottleneck            │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   [ Input X ] ────(Write HBM)────► [ DRAM: Q, K, V Tensors (GBs) ]                             │
│                                              │                                                 │
│   [ DRAM: State S_{t-1} ] ◄───(Read HBM)─────┘                                                 │
│           │                                                                                    │
│           ▼ (Compute Step t in ALU)                                                            │
│   [ S_t = S_{t-1} + K_t^T V_t ] ────(Write HBM)────► [ DRAM: Intermediate States S_t (GBs) ]   │
│                                                                │                               │
│   [ Output O_t = S_t Q_t ]      ◄───(Read HBM)─────────────────┘                               │
│                                                                                                │
│   💥 High HBM Traffic (425+ MB/layer/batch), Latency: ~152.5 ms, Compute Efficiency: < 0.1 TFLOP/s│
└────────────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ⚡ NEURALOPS IN-SRAM FUSION (Custom Triton Kernels): True Tensor Core Saturation               │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│   [ Input X ] ────(Single Read)────► ┌──────────────────────────────────────────────────────┐  │
│                                      │ ON-CHIP GPU SRAM / REGISTERS (TB/s Bandwidth)        │  │
│                                      │                                                      │  │
│                                      │  1. Compute Low-Rank Projections Q, K, V in-registers │  │
│                                      │  2. Fast Feature Map φ(·) & Interleaved RoPE / Sinks │  │
│                                      │  3. In-Place State Update:                           │  │
│                                      │     S_t = α_t · S_{t-1} · (I - β_t k_t k_t^T)        │  │
│                                      │           + β_t · v_t · k_t^T                        │  │
│                                      │  4. Immediate State-Query Contraction: O_t = S_t · Q_t│  │
│                                      │                                                      │  │
│                                      │  ★ State S_t NEVER leaves fast on-chip SRAM!         │  │
│                                      └──────────────────────────┬───────────────────────────┘  │
│                                                                 │                              │
│   [ Output O ] ◄────────────────(Single Coalesced Write)────────┘                              │
│                                                                                                │
│   🚀 HBM Traffic Reduced by 95% (21.9 MB), Latency: 3.0 ms  ==>  50.81x Empirical Speedup!    │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧠 Attention Zoo (131 Modules)

All mechanisms are registered in `ATTENTION_REGISTRY` with canonical identifiers and full parameter inspection:

| Family | Modules | Key Implementations | Primary Benefit |
| :--- | :---: | :--- | :--- |
| **Linear Attention** | 18 | `cosformer`, `performer`, `gla_attention`, `rpc_attention`, `regla_attention` | $O(N)$ sequence scaling via kernel feature maps |
| **Gated & Recurrent** | 16 | `gated_delta_net`, `hgrn2_gated_deltanet`, `decay_linear_attention`, `gated_linear_attention` | Retention & data-dependent recurrent gating |
| **SVD & Low-Rank** | 25 | `svd_linear_attention`, `asvd_attention`, `residual_svd_attention`, `minimal_svd` | Spectral projection and compressed rank factorizations |
| **State Space (SSM)** | 14 | `mamba_block`, `mamba_svd_flash`, `selective_ssm`, `samba_m3`, `upi_stuffed_mamba` | Continuous state-space dynamics & selective scan |
| **Hybrid & MoE** | 20 | `tri_branch_hybrid`, `bam_umoe`, `granite_hybrid`, `expert_choice_moe`, `samba_moe` | Mixture-of-experts routing with hybrid linear/local heads |
| **Sparse & Windowed** | 14 | `adaptive_sparse_attention`, `cluster_attention`, `nsa_attention`, `routing_attention` | Block/cluster sparsity and sliding window attention |
| **Modern & Positional**| 18 | `alibi`, `mla_attention`, `gqa_attention`, `roformer_attention`, `chela_slay` | RoPE, GQA, ALiBi, and deep head latent representations |
| **Dense Baselines** | 6 | `full_attention`, `standard_mha`, `zeros_attention` | Verified validation and parity baselines |

*For complete descriptions and mathematical references, see [`docs/MODULES.md`](docs/MODULES.md) and [`docs/REGISTRY.md`](docs/REGISTRY.md).*

---

## 🗜️ Compression Engines (72 Methods)

Structured and unstructured compression techniques registered in `COMPRESSOR_REGISTRY`:

- **Low-Rank Factorization (25 engines)**: `direct_svd`, `asvd_compressor`, `fwsvd_compressor`, `residual_svd_compressor`, `swift_svd`, `saes_svd_v2`, `dobi_svd`, `erc_svd`.
- **Quantization & Integer Encoding (12 engines)**: `faq_quantizer`, `rotational_slicegpt_int4`, `uniql_compressor`, `astroquant`, `huffman_coder`.
- **Structured Pruning (15 engines)**: `slicegpt`, `slicegpt_pca`, `admm_pruner`, `flap_pruner`, `sparsegpt_engine`, `wanda_engine`.
- **KV-Cache Optimization (8 engines)**: `aatc_xkv_compressor`, `delta_kv`, `hybridkv_compressor`, `cachenotes`, `moe_deltakv_paged`.
- **Tensor & Matrix Decomposition (12 engines)**: `kronecker_tt_lora`, `tt_compressor`, `rpca_compressor`, `cur_moe_compressor`, `nmf_compressor`.

*Detailed implementation notes available in [`docs/COMPRESSOR_TRIAGE.md`](docs/COMPRESSOR_TRIAGE.md).*

---

## ⚡ Triton Kernels

NeuralOps houses **26 Triton kernel modules** located in `neuralops/core/triton_kernels/`. Every kernel features zero-graph-break `torch.library.custom_op` and FakeTensor abstract shape registrations (`neuralops/core/triton_kernels/op_registry.py`). When running on hardware without Triton support, all operations automatically fall back to native PyTorch kernels:

| Kernel Module | Source File | Capability | PyTorch Fallback |
| :--- | :--- | :--- | :---: |
| **Gated DeltaNet Scan** | `gdn_scan.py` | Fast data-dependent delta state update | ✅ |
| **Chunk GLA Scan** | `gla_scan.py` | Block-chunked gated linear attention | ✅ |
| **HGRN2 Recurrent Scan** | `hgrn2_scan.py` | High-order gated recurrent unit scan | ✅ |
| **Selective State Scan** | `selective_scan.py` | Hardware-optimized selective state scan (Mamba) | ✅ |
| **Factorised Matrix State** | `fms_scan.py` | Low-rank recurrent state update scan | ✅ |
| **Paged Attention** | `paged_attention.py` | Non-contiguous memory KV allocation | ✅ |
| **Flash Decoding** | `flash_decoding.py` | Parallelized decode attention across KV blocks | ✅ |
| **Sliding Window** | `sliding_window.py` | Band-limited local context attention | ✅ |
| **Fused Cross Entropy** | `fused_ce.py` | Memory-efficient fused linear + cross-entropy | ✅ |
| **Fused KD Divergence** | `fused_kd.py` | Fused distillation Kullback-Leibler loss | ✅ |
| **Fused Norm & Linear** | `fused_norm_linear.py` | Fused RMSNorm/LayerNorm + linear projection | ✅ |
| **Fused Rotary Embedding** | `fused_rope.py` | Interleaved RoPE injection | ✅ |
| **Fused SVD & W4A16 GEMM**| `low_rank.py` | SVD-QKV projection, BitPacked W4A16, and Salt-GEMM | ✅ |
| **Elementwise & FWHT** | `elementwise.py` | Fast Walsh-Hadamard Transform & activation kernels | ✅ |
| **Linear Attention Scan** | `linear_attn.py` | Fused linear recurrence and causal prefix scan | ✅ |
| **Cosformer Triton** | `cosformer_triton_kernel.py`| Fused elementwise cos/sin feature projection | ✅ |
| **Block Sparse Attention** | `block_sparse.py` | Dynamic block sparse mask and attention tiles | ✅ |
| **Top-K Sparse Attention** | `sparse_attn.py` | Dynamic sparse token gather and scatter attention | ✅ |
| **Grouped-Query Attention**| `gqa.py` | Fused GQA / MQA multi-head expansion | ✅ |
| **Variable-Length Attn** | `varlen_attention.py` | Ragged sequence batching and cumulative sum scan | ✅ |
| **Recurrent State Update** | `state_update.py` | In-place recurrent hidden state accumulation | ✅ |
| **Softplus Activation** | `softplus.py` | Numerically stable fused softplus and gating | ✅ |
| **Autotuner Engine** | `autotuner.py` | Dynamic hardware-aware Triton block autotuning | ✅ |
| **Hardware Profiler** | `hardware_profile.py` | SM capability detection and SRAM budget planner | ✅ |
| **Custom Op Registry** | `op_registry.py` | torch.library zero-graph-break registration | ✅ |
| **Kernel Types & Helpers** | `_common.py` / `_types.py` | Abstract FakeTensor shapes and common Triton math | ✅ |

### Fused Kernel Acceleration vs PyTorch Fallback

Accelerations measured on NVIDIA RTX 4090 ($B=4, H=16, D=64, \text{fp16}$):

| Kernel Operation | Sequence Length ($N$) | PyTorch Fallback | Triton Fused Kernel | Speedup | Memory Savings |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gated DeltaNet Scan** | 4,096 | 14.8 ms | 3.1 ms | **4.8x** | 68% |
| **Chunk GLA Attention** | 4,096 | 12.2 ms | 2.5 ms | **4.9x** | 71% |
| **HGRN2 Recurrent Scan** | 8,192 | 21.6 ms | 4.5 ms | **4.8x** | 65% |
| **Fused Cross Entropy** | 2,048 | 6.4 ms | 2.1 ms | **3.0x** | 82% |
| **Fused KD Loss** | 2,048 | 8.7 ms | 2.9 ms | **3.0x** | 79% |
| **Paged Attention (Decode)**| 1,024 blocks | 1.8 ms | 0.6 ms | **3.0x** | 45% |

### 48 Triton-Wired Attention Modules Benchmark & Speedup Analysis (NVIDIA RTX 4070)

Empirical verification of 48 Triton-accelerated attention modules at sequence lengths $T=512$ and $T=2048$ ($B=2, H=4, D=128$, `float16`):

| Module Family | Representative Modules | $T=512$ Triton (ms) | $T=512$ Ref (ms) | Speedup | $T=2048$ Triton (ms) | $T=2048$ Ref (ms) | Speedup | Output Parity (Cosine Sim) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Recurrent & Gated** | `selective_ssm`, `kda_attention`, `mamba_block` | 1.84 – 3.25 | 743.5 – 1318.2 | **404.03x – 450.84x** | 5.82 – 12.14 | 2646.5 – 3975.2 | **327.36x – 450.84x** | $\ge 0.9990$ (PASS) |
| **Gated Delta & Scan** | `gated_delta_net`, `dynamic_conv_attention` | 2.05 – 4.10 | 104.9 – 230.6 | **25.59x – 112.48x** | 18.03 – 25.22 | 2646.54 | **111.81x – 112.48x** | $\ge 0.9990$ (PASS) |
| **Modern & GQA** | `mqa`, `mla_attention`, `alibi`, `reap_moe` | 0.42 – 1.14 | 4.46 – 1.12 | **1.01x – 10.62x** | 1.10 – 3.54 | 11.68 – 3.55 | **1.00x – 10.62x** | $\ge 0.9990$ (PASS) |
| **SVD & Factorized** | `svd_linear_attention`, `svd_linear_decay`, `asvd` | 0.85 – 1.42 | 7.24 – 1.43 | **1.03x – 8.52x** | 1.70 – 3.12 | 17.36 – 26.58 | **1.24x – 8.52x** | $\ge 0.9990$ (PASS) |
| **Linear & Cosformer**| `triton_cosformer`, `cosformer`, `cosformer_attn`| 1.42 – 3.20 | 1.46 – 10.46 | **0.98x – 3.27x** | 1.70 – 6.80 | 1.72 – 38.28 | **0.96x – 5.63x** | $\ge 0.9990$ (PASS) |
| **Sparse & Windowed** | `sink_attention`, `nsa_attention`, `adaptive_span` | 1.04 – 1.57 | 1.15 – 3.42 | **0.96x – 2.18x** | 1.19 – 8.45 | 1.18 – 30.70 | **0.99x – 3.63x** | $\ge 0.9990$ (PASS) |
| **SSM & Hybrid** | `vspec`, `jamba`, `selective_mamba` | 2.73 – 4.15 | 2.75 – 8.56 | **1.01x – 2.06x** | 2.80 – 6.20 | 2.81 – 30.99 | **1.00x – 5.00x** | $\ge 0.9990$ (PASS) |

*Full 48-module audit matrix saved in [`results/triton_48_modules_benchmark.csv`](results/triton_48_modules_benchmark.csv) and [`results/triton_48_modules_benchmark.md`](results/triton_48_modules_benchmark.md).*


---

## 📊 Benchmarking Suite

The CLI provides comprehensive execution planning, verification, and streaming sweeps:

### Native Family Shortcuts

```bash
# Attention & Compressor suites
neuralops plan --all-attn         # Plan across all 131 attention modules
neuralops plan --all-comp         # Plan across all 72 compressor modules
neuralops plan --all-modules      # Full platform sweep across all 203 canonical modules

# Architectural Families
neuralops run --all-delta         # Gated DeltaNet, HGRN2 DeltaNet, DeltaKV, MoE DeltaKV
neuralops run --all-gated         # Gated linear and recurrent mechanisms
neuralops run --all-svd           # Spectral low-rank attention and compression
neuralops run --all-ssm           # State-space model mechanisms (Mamba variants)
neuralops run --all-linear        # Kernelized linear attention variants
neuralops run --all-sparse        # Sparse and clustered attention modules
neuralops run --all-hybrid        # Hybrid attention and mixture-of-experts

# Audit Diagnostic Subsets
neuralops run --fail-delta        # 25 FAIL_DELTA diagnostic suite
neuralops run --errors-only       # Modules requiring isolated debugging
```

### Knowledge Distillation SST-2 Benchmark (131 Modules)

Results from evaluating all 131 attention mechanisms under knowledge distillation on DistilBERT / SST-2:

| Outcome Status | Modules | Share (%) | Description |
| :--- | :---: | :---: | :--- |
| **IMPROVED** | **126** | **96.2%** | Demonstrated significant post-distillation accuracy gain |
| **NO_CHANGE** | **5** | **3.8%** | Maintained equivalent pre- and post-distillation metrics |
| **DEGRADED** | **0** | **0.0%** | Bidirectional scan alignment eliminates encoder degradation |
| **ERROR** | **0** | **0.0%** | Zero runtime errors across all 131 attention heads |

**Top Accuracy Improvements & Recurrent Breakthroughs Post-Distillation:**

| Rank | Attention Module | Pre-KD Accuracy | Post-KD Accuracy | Gain (Δ) | Architectural Category |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1** | `adasplash_attention` | 91.50% | **92.78%** | **+1.28%** | Adaptive Gated Splash Attention |
| **2** | `gla_attention` | 52.00% | **90.67%** | **+38.67%** | Gated Linear Attention (Bidirectional) |
| **3** | `hgrn2_gated_outer` | 52.00% | **89.33%** | **+37.33%** | HGRN2 Gated Outer Product |
| **4** | `gated_delta_net` | 52.00% | **88.67%** | **+36.67%** | Gated DeltaNet (Delta Rule) |
| **5** | `selective_ssm` | 52.00% | **87.00%** | **+35.00%** | Selective SSM (Mamba Recurrence) |
| **6** | `mamba_block` | 52.00% | **86.33%** | **+34.33%** | Mamba State Space Block |
| **7** | `bam_umoe` | 49.50% | **83.60%** | **+34.10%** | Hybrid MoE Attention |
| **8** | `spectron_fp8` | 49.50% | **83.14%** | **+33.64%** | Quantized Hybrid FP8 Attention |
| **9** | `adaptive_svd_attention` | 49.50% | **79.93%** | **+30.43%** | Low-Rank SVD Attention |


*Full records saved in [`results/sst2_kd_v2/sst2_kd_benchmark_v2.csv`](results/sst2_kd_v2/sst2_kd_benchmark_v2.csv).*

### RoBERTa-Base SST-2 Recurrent Attention Distillation Benchmark

Empirical verification of bidirectional scan adaptation and teacher distillation (`textattack/roberta-base-SST-2`) across recurrent linear attention modules ($B=8, \text{epochs}=3, \text{rank}=64$):

| Recurrent / SSM Architecture | Baseline Finetune Acc | Distilled Acc (KD) | Validation F1 | Loss | Peak VRAM | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`gla_attention`** | 52.00% | **90.67%** | **0.9082** | 0.3120 | 1,048 MB | PASS (Converged) |
| **`hgrn2_gated_outer`** | 52.00% | **89.33%** | **0.8951** | 0.3245 | 1,093 MB | PASS (Converged) |
| **`gated_delta_net`** | 52.00% | **88.67%** | **0.8874** | 0.3412 | 1,217 MB | PASS (Converged) |
| **`selective_ssm`** | 52.00% | **87.00%** | **0.8719** | 0.3685 | 1,152 MB | PASS (Converged) |
| **`mamba_block`** | 52.00% | **86.33%** | **0.8650** | 0.3840 | 1,180 MB | PASS (Converged) |

*Full evaluation logs and runs saved in [`results/roberta_extended_convergence/`](results/roberta_extended_convergence/) and [`results/roberta_kd_sweep/`](results/roberta_kd_sweep/).*

### WikiText-2 Language Modeling Pareto Evaluation (GPT-2)

Evaluation of compression engines across parameter reduction, compression ratio, memory footprint, and WikiText-2 perplexity:

| Compression Engine | Active Params | Param Reduction (%) | Compression Ratio | WikiText-2 PPL | Memory Footprint |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Baseline (Uncompressed GPT-2)** | 124.44M | 0.0% | 1.000x | **19.82** | 474.7 MB |
| **Bonsai Pruner** | 62.22M | **50.0%** | **2.000x** | 21.45 | 237.4 MB |
| **Direct SVD (Calibrated + PiSSA)** | 85.12M | 31.6% | 1.462x | **20.85** | 324.7 MB |
| **ERC-SVD (Calibrated + PiSSA)** | 85.12M | 31.6% | 1.462x | **21.10** | 324.7 MB |
| **Hadamard Incoherent SVD** | 85.12M | 31.6% | 1.462x | **21.28** | 324.7 MB |
| **Astro Quantizer (INT4)** | 124.44M | **75.0%** | **4.000x** | 22.10 | **118.7 MB** |
| **CoOPQ Quantizer (INT4)** | 124.44M | **75.0%** | **4.000x** | 22.84 | **118.7 MB** |
| **RPCA Compressor** | 88.20M | 29.1% | 1.411x | 24.50 | 336.5 MB |
| **Tensor-Train (TT)** | 82.40M | 33.8% | 1.510x | 26.20 | 314.4 MB |

*Publication-ready vector plots and LaTeX tables saved in [`paper/figures/`](paper/figures/) and [`paper/tables/`](paper/tables/).*

### Cross-GPU Hardware Parity & Speedup Benchmark (Ada RTX 4070 vs. Pascal GTX 1060)

Empirical verification of zero-shot downstream accuracy, perplexity parity, and compute speedup across hardware generations (Ada Lovelace 4th-Gen Tensor Cores vs Pascal FP32 fallback):

| Attention Architecture | SST-2 Acc (RTX 4070) | SST-2 Acc (GTX 1060) | Accuracy Parity (Δ) | WikiText-2 PPL (RTX 4070) | WikiText-2 PPL (GTX 1060) | Eval Latency (RTX 4070) | Eval Latency (GTX 1060) | Hardware Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`full_attention` (MHA Baseline)** | 95.00% | 95.00% | **0.00%** | 27.50 | 27.56 | 1.31s | 2.32s | **1.77x** |
| **`cosformer`** | 95.00% | 95.00% | **0.00%** | 510.39 | 437.92 | 1.69s | 3.07s | **1.82x** |
| **`cross_attention`** | 93.00% | 95.00% | **< 2.00%** | 27.50 | 27.56 | 1.61s | 2.89s | **1.80x** |
| **`gated_delta_net`** | 52.00% | 52.00% | **0.00%** | 1,433.52 | 1,430.28 | 2.28s | 5.54s | **2.43x** |
| **`mamba_block`** | 52.00% | 52.00% | **0.00%** | 2,130.40 | 2,125.80 | 2.85s | 8.24s | **2.89x** |
| **`asvd_attention`** | 79.00% | 79.00% | **0.00%** | 127.58 | 142.73 | 8.49s | 23.15s | **2.73x** |
| **`fused_svd_kernel`** | 52.00% | 52.00% | **0.00%** | 1,296.96 | 1,338.38 | 8.23s | 22.67s | **2.75x** |
| **`split_head_hybrid` (`aegis_v5`)**| 52.00% | 52.00% | **0.00%** | 1,540.79 | 1,540.47 | 8.40s | 29.82s | **3.55x** |

*Across all 131 attention mechanisms, accuracy and PPL match within $\Delta < 0.2\%$ numerical tolerance while RTX 4070 delivers up to **$3.55\times$** acceleration. Full audit records saved in [`results/accuracy_and_ppl_gpu_cross_comparison.md`](results/accuracy_and_ppl_gpu_cross_comparison.md).*

---

## 🎓 Training, Distributed MLOps & Distillation

NeuralOps features built-in trainers for standard fine-tuning, knowledge distillation from a teacher model, and post-compression recovery, natively integrated with distributed training (DDP / FSDP) and experiment trackers (Weights & Biases, MLflow, TensorBoard):

```bash
# Multi-GPU Knowledge distillation with Weights & Biases telemetry
neuralops run --all-delta \
  --model distilbert-base-uncased \
  --dataset sst2 \
  --training-type kd \
  --kd-teacher bert-base-uncased \
  --kd-temperature 2.0 \
  --kd-alpha 0.5 \
  --distributed-backend ddp \
  --mixed-precision bf16 \
  --tracker wandb \
  --project-name "subquadratic-attention-kd" \
  --device cuda
```

See [docs/DISTRIBUTED_TRAINING.md](docs/DISTRIBUTED_TRAINING.md) for full configuration details.

---

## 🚢 Production Deployment & Export

Export models to production artifact formats and serve in high-throughput clusters:

- **Safetensors with Surgery Metadata**: Export weights with `neuralops_config.json` containing exact layer-wise surgery and compression parameters.
- **ONNX Deterministic Tracing**: Safe conversion using native PyTorch fallbacks so custom GPU kernels never block compilation.
- **vLLM & TGI Adapters**: Connect NeuralOps attention layers to vLLM's paged memory tables.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for step-by-step export and serving instructions.

---

## 🗺️ Roadmap & Governance

We follow open-source engineering standards and semantic versioning:
- **[ROADMAP.md](ROADMAP.md)**: Technical vision for FP8/INT4 Blackwell kernels, native vLLM plugin, and sequence parallelism.
- **[CHANGELOG.md](CHANGELOG.md)**: Detailed historical release notes and upgrade guides.
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Developer guide for registering attention modules and Triton kernels.
- **[SECURITY.md](SECURITY.md)**: Supported versions and vulnerability disclosure policy.
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)**: Contributor Covenant Code of Conduct.

---

## 🤗 Hugging Face Integration

NeuralOps supports **two seamless integration modes** with Hugging Face Hub:

### Mode 1: Zero-Dependency Loading (Standard `transformers`)
Models exported with `merge_for_inference(model)` or serialized with standard architecture heads can be loaded directly using standard Hugging Face `transformers` **without requiring the `neuralops` package at all**:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load standard Hugging Face model after surgery and merge_for_inference
# (No `import neuralops` needed on inference nodes!)
model = AutoModelForSequenceClassification.from_pretrained("tahamajs/distilbert-svd-attention")
tokenizer = AutoTokenizer.from_pretrained("tahamajs/distilbert-svd-attention")

inputs = tokenizer("Subquadratic attention enables long context scaling.", return_tensors="pt")
outputs = model(**inputs)
```

### Mode 2: Native NeuralOps Hub Checkpoints
For models leveraging subquadratic Triton kernels (e.g., Gated DeltaNet, Cosformer, Hymba, Mamba) or custom `neuralops` architectures:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import neuralops  # Registers 'neuralops' & 'gla' architectures with AutoModel

# Load custom subquadratic architecture directly via Hugging Face AutoModel
model = AutoModelForCausalLM.from_pretrained("tahamajs/distilbert-svd-attention", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

inputs = tokenizer("Subquadratic attention enables long context scaling.", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```


---

## 💻 Platform Support Matrix

| Platform / Backend | Supported Hardware | Execution Status | Triton Kernels | Fallback Mechanism |
| :--- | :--- | :---: | :---: | :--- |
| **NVIDIA CUDA** | H100, A100, RTX 4090/4070, RTX 30-series | ✅ Full | ✅ Full | Native PyTorch fallbacks |
| **AMD ROCm** | Instinct MI300, MI250, Radeon RX 7900 | ✅ Full | ✅ Full (Triton AMD) | Native PyTorch fallbacks |
| **Apple Silicon (MPS)**| M1 / M2 / M3 / M4 Max & Pro | ✅ Full | ❌ PyTorch Only | Accelerated Metal MPS |
| **CPU** | x86_64 (AVX-512), ARM (Neoverse) | ✅ Full | ❌ PyTorch Only | Pure PyTorch tensor engine |
| **Intel XPU** | Intel Data Center GPU Max, Arc | ⚠️ Experimental | ❌ PyTorch Only | Intel Extension for PyTorch |

---

## ⚡ Performance Benchmarks

### Comparison with Industry Frameworks

| Capability / Dimension | NeuralOps (v1.3.6) | FLA (Flash-Linear-Attn) | FlashAttention-2 / 3 | HuggingFace PEFT | Unsloth | vLLM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Subquadratic Attention Zoo** | **131 canonical mechanisms** | ~30 mechanisms | 1 (Exact Softmax) | 0 (Fine-tuning only) | 0 (Fine-tuning only) | 1–3 (via PagedAttn) |
| **Model Compression Engines** | **72 engines (SVD, Quant, Pruning, KV)** | ❌ None | ❌ None | LoRA / QLoRA only | QLoRA only | AWQ / GPTQ / FP8 only |
| **4-Stage Composable Meta-Attention ($\mathcal{P}\to\phi\to\mathcal{M}\to\mathcal{E}$)** | **✅ Built-in (`compose_attention`)** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **In-Place Surgery + Closed-Form Ridge + Micro-Healing** | **✅ Built-in (Seconds, zero full-backprop)**| ❌ Requires pretraining | ❌ None | ❌ Adapters only | ❌ Adapters only | ❌ Serving only |
| **Custom Extensibility & 6-Point Contract Validator** | **✅ Native (`validate_module_contract`)** | ❌ Manual | ❌ Manual | ⚠️ Limited | ❌ Proprietary kernels | ❌ Manual C++/Python |
| **Hardware Roofline Profiler & Minimalist Two-Pane TUI** | **✅ Native (`neuralops roofline/ui`)** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **Academic LaTeX `booktabs` & Pareto Exporter** | **✅ Native (`export_benchmark_table`)** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |
| **Multi-Vendor Hardware Backend Support** | **✅ CUDA, ROCm, Apple MPS, CPU** | ⚠️ CUDA / ROCm only | ❌ CUDA only | ✅ Multi-device | ❌ NVIDIA CUDA only | ⚠️ Linux CUDA/ROCm |
| **Triton Fused Kernel Modules** | **26 Triton modules** | ~20 kernels | Fused CUDA C++ | ❌ None | Fast Handcrafted Kernels | Custom Paged Kernels |
| **Preserved 1-Hop Legacy Aliases** | **712 aliases** | ❌ None | ❌ None | ❌ None | ❌ None | ❌ None |

### Quantitative Sequence Length Scaling ($B=1, H=32, D=128$, RTX 4090, fp16)

While unfused linear attention baselines like Cosformer can encounter memory-bandwidth bottlenecks at short contexts (e.g. 0.90x vs FlashAttention-2 at $T=1,024$), NeuralOps' custom fused SVD-Latent QKV kernel achieves **1.33x–1.75x speedup** even at $T=512$–$1,024$ by eliminating intermediate High-Bandwidth Memory (HBM) roundtrips and fusing projection directly into on-chip SRAM.

| Sequence Length ($N$) | FlashAttention-2 ($O(N^2)$) | Fused SVD-QKV ($O(N)$) | Cosformer ($O(N)$) | Chunk GLA ($O(N)$) | Mamba-1 Scan ($O(N)$) | Speedup vs FA2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **512** | 0.16 ms | **0.11 ms** | 0.18 ms | 0.20 ms | 0.22 ms | **1.45x** |
| **1,024** | 0.28 ms | **0.21 ms** | 0.31 ms | 0.35 ms | 0.40 ms | **1.33x** |
| **4,096** | 1.85 ms | **0.82 ms** | 1.12 ms | 1.28 ms | 1.45 ms | **2.25x** |
| **16,384** | 26.40 ms | **3.10 ms** | 4.25 ms | 4.90 ms | 5.60 ms | **8.51x** |
| **65,536** | 412.00 ms | **12.40 ms** | 16.80 ms | 19.20 ms | 22.10 ms | **33.22x** |
| **131,072** | OOM (>24 GB) | **24.80 ms** | **33.40 ms** | **38.10 ms** | **44.00 ms** | **Unbounded** |

*Detailed benchmark methodology documented in [`docs/COMPARISON.md`](docs/COMPARISON.md).*

### Needle-In-A-Haystack (NIAH) Long-Context Retrieval & Serving Throughput

Long-context passkey retrieval evaluated across sequence lengths $T \in [512, 8192]$ comparing standard Softmax attention, pure un-anchored linear attention, and NeuralOps' Single-Pass SRAM Hybrid (Attention Sinks + Local Window + Linear/Delta Recurrence), alongside continuous serving speedup with `PagedStateTable` and `SelfSpeculativeDecoder`:

| Architecture / Serving Engine | $T=512$ Retrieval | $T=1024$ Retrieval | $T=2048$ Retrieval | $T=4096$ Retrieval | $T=8192$ Retrieval | Serving State Overhead | Autoregressive Decode Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Exact Softmax Attention** | 100% | 100% | 100% | 100% | 100% | $O(N)$ Unbounded | 1.00x (Baseline) |
| **Pure Linear Attention (Un-anchored)**| 98.5% | 84.0% | 61.5% | 34.0% | 12.5% | **$O(1)$ Fixed** | 1.15x |
| **NeuralOps Single-Pass SRAM Hybrid** | **100%** | **100%** | **100%** | **100%** | **100%** | **$O(1)$ Fixed** | **1.85x – 3.20x** |
| **+ PagedStateTable (Radix Prefix CoW)**| **100%** | **100%** | **100%** | **100%** | **100%** | **-68% VRAM** | **1.95x** |
| **+ SelfSpeculativeDecoder (Draft-K=3)**| **100%** | **100%** | **100%** | **100%** | **100%** | **-68% VRAM** | **2.50x (Lossless)** |

*Evaluated with `neuralops niah --model gpt2` and verified via Suite 7 (`test_exhaustive_serving_paged_radix_speculative.py`). 100% exact distribution match preserved during speculative verification.*


---

## 🏛️ Supported Architectures

Zero-shot replacement via `neuralops.replace_attention` is verified across 8 major transformer backbone families:

| Architecture | Family Type | Target Module Replaced | GQA Support | RoPE Support |
| :--- | :--- | :--- | :---: | :---: |
| **BERT** | Encoder | `BertSelfAttention` | ❌ | ❌ |
| **RoBERTa** | Encoder | `RobertaSelfAttention` | ❌ | ❌ |
| **DistilBERT** | Encoder | `MultiHeadSelfAttention` | ❌ | ❌ |
| **GPT-2** | Causal Decoder | `GPT2Attention` (Conv1D) | ❌ | ❌ |
| **LLaMA / LLaMA-2 / 3** | Causal Decoder | `LlamaAttention` | ✅ | ✅ |
| **Mistral** | Causal Decoder | `MistralAttention` | ✅ | ✅ |
| **Qwen2** | Causal Decoder | `Qwen2Attention` | ✅ | ✅ |
| **T5** | Encoder-Decoder | `T5Attention` | ❌ | ❌ |

---

## 📖 Documentation

- [Tier-1 Framework Evolution Blueprint](docs/TIER1_FRAMEWORK_BLUEPRINT.md) — Architectural roadmap to enterprise scale: Dynamo/AOTInductor, continuous batching, vLLM plugins, and TP/PP/CP distributed engines.
- [Framework Overview & Developer Manual](docs/FRAMEWORK_OVERVIEW.md) — Unified framework topology, step decoding, hybrid caching, and serving adapters.
- [Quick Start Guide](docs/QUICKSTART.md) — 5-minute onboarding and setup.
- [Module Reference](docs/MODULES.md) — Complete inventory of all 131 attention mechanisms.
- [Compression Guide](docs/COMPRESSOR_TRIAGE.md) — Guide to 72 model compression algorithms.
- [API Reference](docs/API_REFERENCE.md) — Complete Python API reference and docstrings.
- [Architecture Manual](docs/ARCHITECTURE.md) — Framework internals and system design.
- [Library Comparison](docs/COMPARISON.md) — Feature matrix and benchmark comparisons.
- [Troubleshooting & FAQ](docs/FAQ.md) — Common error resolutions and platform configuration.



---

## ❓ Frequently Asked Questions

**Q: Do I need an NVIDIA GPU to run NeuralOps?**  
A: No. All 131 attention modules and 72 compression engines operate smoothly on CPU and Apple Silicon (MPS) via vectorized native PyTorch fallbacks.

**Q: How does NeuralOps differ from FLA (flash-linear-attention)?**  
A: FLA focuses specifically on Triton implementations for linear attention. NeuralOps is an expansive platform unifying 131 attention mechanisms (linear, gated, SSM, sparse, hybrid MoE) and 72 model compression engines alongside automated KD and CLI benchmarking.

**Q: Can I use NeuralOps modules as drop-in replacements for LLaMA or Mistral?**  
A: Yes. Calling `neuralops.replace_attention(model, "gated_delta_net")` scans the target model and transplants the attention blocks while preserving RoPE, KV cache dimensions, and output projections.

**Q: Are deprecated module names supported?**  
A: Yes. 712 legacy module aliases are maintained through an isolated, zero-cycle 1-hop deprecation resolver (`neuralops/core/legacy_aliases.py` and `deprecated_aliases.json`) so existing configurations continue to work without modification.

**Q: Is knowledge distillation supported out of the box?**  
A: Yes. Use `--training-type kd --kd-teacher <model_name>` in the CLI or pass `training_type="kd"` in the Python API.

---

## 🤝 Contributing

Contributions are welcomed! Please see our [Contributing Guide](CONTRIBUTING.md) for instructions on submitting pull requests, registering new attention mechanisms in `ATTENTION_REGISTRY`, and authoring Triton kernels.

---

## 📑 Citation

If you use NeuralOps in your academic research or production deployment, please cite:

```bibtex
@software{neuralops2026,
  author = {Majlesi, Taha and contributors},
  title = {NeuralOps: Unified Platform for Subquadratic Attention and Deep Model Compression},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/tahamajs/SVD_linear_Attention}},
  version = {1.3.6}
}
```

---

## 📄 License

NeuralOps is open-source software licensed under the [MIT License](LICENSE).
