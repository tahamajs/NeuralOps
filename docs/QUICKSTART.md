# ⚡ NeuralOps Quickstart Guide (v1.6.0)

Welcome to **NeuralOps** (`SVD_linear_Attention` **v1.6.0**), the unified acceleration library for subquadratic sequence modeling, in-place neural attention surgery, and low-rank model compression.

[![Release](https://img.shields.io/badge/release-v1.6.0-blue.svg)](https://github.com/tahamajs/NeuralOps/releases/tag/v1.6.0)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark%20Ledger-Hooshaai%2Fneuralops--results-ffcc00)](https://huggingface.co/datasets/Hooshaai/neuralops-results)

---

## Canonical Invariants at a Glance

| Ecosystem Component | Verified Canonical Count | Architectural Scope |
| :--- | :---: | :--- |
| **Attention Mechanisms** | **131** | Across 14 architectural families (Dense, Linear, Gated, Delta, SSM, Sparse, SVD, MLA, Hybrid, MoE, TTT, Memory, Positional, Fused) |
| **Compression Engines** | **72** | Across 8 algorithmic paradigms (Low-rank SVD, Dynamic rank, Structured pruning, Subspace, Tensor decomp, Quantization, KV-cache, Rotational invariance) |
| **Total Unified Modules** | **203** | 131 Attention mechanisms + 72 Compression engines verified on physical hardware |
| **Triton JIT Kernel Modules**| **26** | 62 unified operators across 4 architectural tiers with automated hardware autotuning |
| **Native CUDA C++ Packages** | **112** | 114 native NVCC kernel implementations with warp shuffles and MMA intrinsics |
| **Master Jupyter Notebooks** | **29** | Reproducible demonstration tutorials (00–25, empirical master, integration suites) |

> [!NOTE]
> Developed at the **Department of Computer Engineering, College of Engineering, University of Tehran**.  
> Grounded in physical silicon execution on **NVIDIA GeForce RTX 4070 (Ada Lovelace SM 8.9, AD104 silicon: 5,888 CUDA cores, 46 SMs, 184 4th Gen Tensor Cores, 160.2 TFLOPS boost / 141.6 base FP16, 504.0 GB/s GDDR6X, Arithmetic Intensity Ridge Point $I^* = 317.9\text{ FLOP/Byte}$, 12 GB VRAM)**.  
> All empirical results and benchmark records are mirrored in the official Hugging Face benchmark repository:  
> 👉 **[https://huggingface.co/datasets/Hooshaai/neuralops-results](https://huggingface.co/datasets/Hooshaai/neuralops-results)**

---

## Installation

### Standard Installation via PyPI

```bash
pip install neuralops
```

### Installation from Source (Recommended for Hardware Acceleration)

```bash
git clone https://github.com/tahamajs/NeuralOps.git
cd NeuralOps
pip install -e .
```

### Compiling Native CUDA C++ Extensions (Level 1 Dispatch)

To enable native Level-1 CUDA acceleration (for NVIDIA GPUs SM $\ge$ 7.0 / Ampere, Ada Lovelace, Hopper), install from the repository root with native compilation, or install pre-compiled binary packages:

```bash
# Build extensions in-place from repository root (sources in neuralops/core/cuda_kernels/ and neuralops/ops/cuda/)
pip install -e .

# Or install standalone binary distribution
pip install neuralops-cuda
```

---

## 1. One-Line In-Place Attention Surgery

Transplant standard quadratic attention with subquadratic linear attention or gated state-space models without re-architecting your model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import neuralops

# 1. Load any pre-trained Hugging Face causal LM
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 2. In-place surgery: replace quadratic attention with Gated DeltaNet
student_model = neuralops.replace_attention(
    model,
    attention="gated_delta_net",
    calibrate=True,  # Automatically runs forward calibration
)

# 3. Model retains native Hugging Face AutoModel API
inputs = tokenizer("Subquadratic attention scales linearly with sequence length:", return_tensors="pt")
outputs = student_model(**inputs)
print(f"Surgery successful! Logits shape: {outputs.logits.shape}")
```

---

## 2. Low-Rank Matrix Compression

Compress heavy projection matrices using Fisher-Weighted SVD (FWSVD) or Adaptive SVD:

```python
import neuralops

# Compress model linear layers to target rank r=32
compressed_model = neuralops.compress(
    student_model,
    method="fwsvd",
    rank=32,
)

orig_params = sum(p.numel() for p in model.parameters()) / 1e6
comp_params = sum(p.numel() for p in compressed_model.parameters()) / 1e6
print(f"Original: {orig_params:.1f}M params -> Compressed: {comp_params:.1f}M params")
```

---

## 3. Post-Surgery Distillation Healing

Recover full task accuracy through NeuralOps' 6-pillar knowledge distillation pipeline:

```python
from transformers import TrainingArguments
from neuralops.trainer import NeuralOpsTrainer

training_args = TrainingArguments(
    output_dir="./healed_checkpoint",
    learning_rate=5e-5,
    max_steps=500,
    per_device_train_batch_size=8,
)

trainer = NeuralOpsTrainer(
    model=compressed_model,
    teacher_model=model,
    kd_loss=True,
    kd_temperature=2.0,
    kd_weight=0.5,
    args=training_args,
)
trainer.train()
```

---

## 4. 3-Level Fallback Kernel Dispatch

NeuralOps automatically routes kernel operations through the optimal execution path:

```python
import torch
from neuralops import force_kernel_backend
from neuralops.core.backend.dispatch import available_backends, get_backend

# Check active dispatch backend
backends = available_backends()
backend = get_backend()
print(f"Optimal backend selected: {backend}")  # e.g., 'triton' -> 'cuda' -> 'cpu'

# Force specific backend for verification or ablation
with force_kernel_backend("triton"):
    from neuralops.ops.triton.gdn_scan import gdn_scan_triton
    q = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 8, 512, 64, device="cuda", dtype=torch.float16)
    beta = torch.sigmoid(torch.randn(2, 8, 512, device="cuda", dtype=torch.float16))
    alpha = torch.sigmoid(torch.randn(2, 8, 512, device="cuda", dtype=torch.float16))
    out = gdn_scan_triton(q, k, v, beta, alpha)
```

---

## 5. High-Throughput Serving & Continuous Batching

NeuralOps provides production serving adapters for vLLM, Text Generation Inference (TGI), and SGLang:

```python
from neuralops.integrations.tgi import TGIServingAdapter

# Initialize continuous batching adapter with request queue
serving_engine = TGIServingAdapter(
    model=compressed_model,
    tokenizer=tokenizer,
    max_batch_size=16,
)

# Stream generation tokens incrementally
prompt = "Explain associative memory:"
for token_id, token_text in serving_engine.streaming_generate(prompt, max_new_tokens=30):
    print(f"Token {token_id}: {token_text}", end="", flush=True)
print()
```

---

## 6. Reproducible Jupyter Notebooks (29 Master Tutorials)

The repository includes **29 master research notebooks** located in `notebooks/`:

| Notebook | Title | Primary Topic |
| :---: | :--- | :--- |
| `00` | Complete System Walkthrough | In-place surgery, registry inspection, roundtrip export |
| `01` | Quickstart Surgery and Healing | End-to-end DistilBERT surgery on SST-2 |
| `02` | Roofline & Triton Kernels | Arithmetic intensity profiling on Ada Lovelace |
| `05A`| Master 108 CUDA & 26 Triton Benchmark | Micro-benchmarking all native CUDA and Triton kernels |
| `06` | Master All 131 Attention Zoo | Interactive showcase of 131 attention mechanisms |
| `07` | Master All 72 Compression Engines | Benchmark of SVD, pruning, and quantization |
| `16` | Roofline Ada RTX 4070 Saturation | Empirical hardware roofline on SM 8.9 |
| `24` | Comprehensive Empirical Master | Full 203-module evaluation matrix and Pareto curves |

---

## Next Steps

- Consult the step-by-step **[Getting Started Guide](guides/getting_started.md)**, **[Attention Surgery Guide](guides/attention_surgery.md)**, **[Model Compression Guide](guides/model_compression.md)**, and **[Production Serving Guide](guides/production_serving.md)**.
- Explore the **[API Reference](api/attentions.md)** for exhaustive breakdowns of all 131 attentions across 14 families and 72 compressors across 8 paradigms.
- Test the interactive **[Ada Roofline Profiler](roofline_interactive.html)** in your browser.
- Open the live **[Telemetry Dashboard](dashboard.html)** for real-time GPU metrics.
- Inspect raw benchmark runs in the **[Hugging Face Results Repository](https://huggingface.co/datasets/Hooshaai/neuralops-results)**.
