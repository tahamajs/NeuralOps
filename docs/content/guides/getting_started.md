# Getting Started with NeuralOps

Welcome to **NeuralOps** (`SVD_linear_Attention` **v1.6.0**), an open-source, publication-grade acceleration and model compression library developed at the **Department of Computer Engineering, College of Engineering, University of Tehran**.

NeuralOps bridges the gap between theoretical subquadratic sequence models and real-world high-throughput production deployment. It combines:

1. **131 Attention Mechanisms** across **14 Architectural Families**.
2. **72 Model Compression Engines** across **8 Paradigms**.
3. **26 Canonical Triton Kernel Modules** (62 unified operators across 4 tiers).
4. **112 Native CUDA C++ Packages** (114 native implementations).
5. **29 Master Demonstration Jupyter Notebooks**.
6. **Zero-Overhead 3-Level Fallback Dispatch** (`CUDA C++` $\to$ `Triton JIT` $\to$ `Vectorized PyTorch`).
7. **Empirically Grounded Physical Hardware Telemetry** benchmarked on **NVIDIA GeForce RTX 4070 (Ada Lovelace SM 8.9, AD104 silicon)**.

[![Release](https://img.shields.io/badge/release-v1.6.0-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v1.6.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

---

## 1. System Requirements & Hardware Target

NeuralOps is designed to run universally across environments while maximizing physical hardware acceleration:

| Component | Minimum Specification | Recommended Production Target |
| :--- | :--- | :--- |
| **GPU Hardware** | Any CUDA GPU (SM $\ge$ 7.0) or CPU/Apple Silicon (MPS) | **NVIDIA GeForce RTX 4070 (SM 8.9 Ada Lovelace)** or **NVIDIA H100 (SM 90 Hopper)** |
| **Silicon Specs** | N/A (Level 3 PyTorch fallback) | AD104 silicon: 5,888 CUDA cores, 46 SMs, 184 4th Gen Tensor Cores, 12 GB GDDR6X |
| **Operating System** | Linux (Ubuntu 20.04+), macOS 13+, Windows WSL2 | Linux (Ubuntu 22.04 LTS / RHEL 9) |
| **Python** | Python 3.10, 3.11, or 3.12 | Python 3.11 |
| **CUDA Toolkit** | CUDA 11.8+ (for Level 1/2) | CUDA 12.4+ / 12.8 with NVCC |
| **PyTorch** | PyTorch 2.1.0+ | PyTorch 2.4.0+ / 2.6.0+ |
| **Triton** | Triton 2.1.0+ (optional) | Triton 3.0.0+ |

All empirical benchmark records and raw telemetry traces are openly hosted on Hugging Face:  
👉 **[https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)**

---

## 2. Installation

### Option A: Standard PyPI Installation

```bash
pip install neuralops
```

### Option B: Editable Development Installation

Clone the repository and install in editable mode with development dependencies:

```bash
git clone https://github.com/tahamajs/NeuralOps.git
cd NeuralOps
pip install -e ".[dev]"
```

### Option C: Compiling Native CUDA C++ Extensions (Level 1 Dispatch)

To unlock Level 1 peak throughput on NVIDIA GPUs:

```bash
cd neuralops/csrc
python setup.py build_ext --inplace
cd ../..
```

---

## 3. Five-Minute Quick Tour

### Step 1: Inspecting Registries

NeuralOps provides lazy discovery and inspection of all 203 unified modules:

```python
import neuralops

# List registered mechanisms
attentions = neuralops.list_attentions()
compressors = neuralops.list_compressors()

print(f"Registered Attention Mechanisms: {len(attentions)}")  # Exactly 131
print(f"Registered Compression Engines:   {len(compressors)}") # Exactly 72
```

### Step 2: One-Line High-Level Attention Surgery (`neuralops.transplant`)

Surgically transplant quadratic attention mechanisms with subquadratic linear attention or recurrent state-space models in-place, complete with automated architecture scanning, closed-form Ridge calibration, and post-surgery micro-healing:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import neuralops

model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# One-line surgical transplant with closed-form calibration & micro-healing
surged_model = neuralops.transplant(
    model=model,
    target_attention="chunk_gla",
    calibration_dataset="fineweb-edu",
    layer_budget="hybrid_sandwich",
)

# Direct forward inference (subquadratic compute & O(1) recurrent state)
inputs = tokenizer("Subquadratic sequence modeling achieves:", return_tensors="pt")
outputs = surged_model(**inputs)
print(f"Transplant complete! Output logits shape: {outputs.logits.shape}")
```

*(For granular, manual low-level replacement, `neuralops.replace_attention(model, attention="cosformer", copy_weights=True)` remains fully available).*

### Step 3: Low-Rank Matrix Compression

Compress model projection matrices using Fisher-Weighted SVD (FWSVD):

```python
# Compress linear layers down to rank r=32
compressed_model = neuralops.compress(
    model=surged_model,
    method="fwsvd",
    rank=32,
)

orig_params = sum(p.numel() for p in model.parameters()) / 1e6
comp_params = sum(p.numel() for p in compressed_model.parameters()) / 1e6
print(f"Parameters: {orig_params:.1f}M -> {comp_params:.1f}M ({orig_params / comp_params:.2f}x compression)")
```

### Step 4: 3-Level Fallback Kernel Execution

Execute optimized Triton or CUDA kernels with automatic PyTorch CPU fallback:

```python
from neuralops.core.dispatch import get_optimal_backend, force_kernel_backend
from neuralops.core.triton_kernels.gdn_scan import gdn_scan_triton

# Determine available execution tier
backend = get_optimal_backend("gdn_scan")
print(f"Selected backend: {backend}")

# Explicitly test Triton backend
with force_kernel_backend("triton"):
    q = torch.randn(2, 8, 512, 64, device="cuda" if torch.cuda.is_available() else "cpu")
    k = torch.randn(2, 8, 512, 64, device="cuda" if torch.cuda.is_available() else "cpu")
    v = torch.randn(2, 8, 512, 64, device="cuda" if torch.cuda.is_available() else "cpu")
    beta = torch.sigmoid(torch.randn(2, 8, 512, device="cuda" if torch.cuda.is_available() else "cpu"))
    out = gdn_scan_triton(q, k, v, beta)
    print("Kernel output shape:", out.shape)
```

### Step 5: Zero-Config Production Serving Export (`neuralops.export`)

Export transplanted or compressed models for high-throughput serving systems (vLLM, SGLang, TensorRT-LLM) with guaranteed zero-fake artifact serialization and cryptographic provenance:

```python
# Zero-config export for native vLLM serving with PagedState tables
export_path = neuralops.export(
    model=compressed_model,
    engine="vllm",
    path="./exported_model",
    tokenizer=tokenizer,
)

print(f"Model exported cleanly to: {export_path}")
# The exported directory includes auto_map, configuration, and weights:
# Launch directly via vLLM:
# vllm serve ./exported_model --trust-remote-code
```

---

## 4. Next Guides

- **[Attention Surgery Guide](attention_surgery.md)**: Master the 5 topological patterns and in-place transplant pipeline.
- **[Model Compression Guide](model_compression.md)**: Explore the 8 compression paradigms and Pareto frontiers.
- **[Kernel Dispatch Guide](kernel_dispatch.md)**: Understand the 3-level fallback hierarchy and custom operator registration.
- **[Production Serving Guide](production_serving.md)**: Deploy models with vLLM PagedState, TGI continuous batching, and SGLang.
- **[Interactive Visualizers](../visualizer/index.html)**: Explore the 3D Attention Lab, Latent Manifold, and Ada Roofline in your browser.
