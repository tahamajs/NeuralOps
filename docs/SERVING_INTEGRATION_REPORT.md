# NeuralOps Serving Integration & Benchmark Report

This document reports the empirical validation and cross-engine serving benchmark results for NeuralOps across PyTorch, vLLM, SGLang, and TensorRT-LLM on physical dual NVIDIA Tesla T4 GPUs.

---

## 1. Environment & Hardware Specifications

- **Host Platform**: Kaggle GPU Environment (Remote Linux Host via SSH)
- **GPUs**: 2x NVIDIA Tesla T4 (15,360 MiB VRAM each, Compute Capability 7.5 Turing)
- **Driver / CUDA Version**: NVIDIA Driver 580.178.04, CUDA 13.0
- **Operating System**: Linux x86_64 (Kernel 6.6.137+)
- **Python**: Python 3.13.15
- **PyTorch**: 2.13.0+cu130
- **vLLM**: 0.32.0 (Installed & Operational)
- **SGLang**: 0.5.23 (Installed; runtime kernel requires SM75 C++ compilation)
- **TensorRT-LLM**: Not installed (DEFERRED: no prebuilt binary wheels available for Python 3.13 / CUDA 13.0 on PyPI)

---

## 2. Per-Engine Integration Status

| Backend | Status | Notes |
|---|---|---|
| **PyTorch (Baseline)** | **AVAILABLE** | Native model surgery via `replace_attention()`. Validated across `cosformer`, `gla_attention`, and `decay_linear_attention`. Fast single-stream execution. |
| **vLLM** | **AVAILABLE BUT SLOWER** | Runs on T4 but ~2x to 4.7x slower than PyTorch baseline. Scaffold code written and verified to run, but not production-ready for latency-sensitive serving on T4. |
| **SGLang** | **NOT COMPATIBLE** | SM75 Turing binaries missing from `sgl_kernel 0.5.23` (`common_ops.abi3.so` symbol mismatch). Compatible only on Hopper (SM90) and Blackwell (SM100). |
| **TensorRT-LLM** | **DEFERRED** | No Python 3.13 Linux wheel on PyPI. NVIDIA requires Python 3.10 or 3.12. Integration plugin unit tests pass offline, engine execution deferred. |

---

## 3. Serving Benchmark Results

Empirical cold/warm-cache benchmarks conducted using `scripts/benchmark_serving.py` with `torch.cuda.Event` high-resolution hardware timers on model `gpt2` at sequence length 512.

```
===============================================================================================
                      NEURALOPS SERVING BENCHMARK RESULTS
===============================================================================================
| Engine         | Attention        | Seq Len   | Latency (ms)   | Throughput (tok/s)   | Peak VRAM (MB)   | Status               |
|----------------|------------------|-----------|----------------|----------------------|------------------|----------------------|
| pytorch        | cosformer        | 512       | 124.39         | 4116.19              | 617.04           | PASS                 |
| pytorch        | gla_attention    | 512       | 90.08          | 5683.58              | 1191.25          | PASS                 |
| pytorch        | decay_linear_attention | 512 | 46.22          | 11078.12             | 1195.60          | PASS                 |
| vllm           | cosformer        | 512       | 209.09         | 2525.24              | 106.91           | AVAILABLE BUT SLOWER |
| vllm           | gla_attention    | 512       | 214.13         | 2465.82              | 106.91           | AVAILABLE BUT SLOWER |
| vllm           | decay_linear_attention | 512 | 220.59         | 2393.63              | 106.91           | AVAILABLE BUT SLOWER |
| sglang         | cosformer        | 512       | N/A            | N/A                  | N/A              | NOT COMPATIBLE       |
| sglang         | gla_attention    | 512       | N/A            | N/A                  | N/A              | NOT COMPATIBLE       |
| sglang         | decay_linear_attention | 512 | N/A            | N/A                  | N/A              | NOT COMPATIBLE       |
| tensorrt_llm   | cosformer        | 512       | N/A            | N/A                  | N/A              | DEFERRED             |
| tensorrt_llm   | gla_attention    | 512       | N/A            | N/A                  | N/A              | DEFERRED             |
| tensorrt_llm   | decay_linear_attention | 512 | N/A            | N/A                  | N/A              | DEFERRED             |
===============================================================================================
```

---

## 4. Interpretation of Benchmark Results

The benchmark numbers provide clear architectural takeaways regarding high-throughput serving runtimes:

1. **vLLM Backend Execution vs. Latency**: The vLLM custom backend runs end-to-end and successfully generates tokens, but it is significantly slower than the PyTorch baseline on dual T4 GPUs:
   - `cosformer`: 209.09 ms (vLLM) vs. 124.39 ms (PyTorch) — **1.7x slower**
   - `gla_attention`: 214.13 ms (vLLM) vs. 90.08 ms (PyTorch) — **2.4x slower**
   - `decay_linear_attention`: 220.59 ms (vLLM) vs. 46.22 ms (PyTorch) — **4.8x slower**
2. **Root Cause Analysis**:
   - `enforce_eager=True` was required on T4, which disables CUDA Graph capture and introduces kernel launch overhead per token.
   - vLLM's PagedAttention and continuous batching scheduling engine incur non-trivial fixed infrastructure overhead per request. At short sequence lengths (512 tokens), this scheduling overhead completely dominates the kernel execution time.
   - For long context windows ($\ge 4096$ tokens) or large batch sizes, vLLM's memory paging is expected to break even or outperform unpaged PyTorch allocations. However, current tests were scoped to 512 tokens on small models (`gpt2`).
   - This is not a defect in vLLM; it is an expected architectural characteristic when deploying small models on older Turing hardware with eager execution.
3. **Serving Recommendations**:
   - For latency-critical production deployments of models $< 1\text{B}$ parameters on T4, **use the PyTorch baseline** with custom CUDA C++ / Triton kernels.
   - On modern Hopper (SM90) or Blackwell (SM100) clusters, evaluate **SGLang** with precompiled native RadixAttention kernels.
   - **Do NOT deploy the vLLM eager-mode backend on T4 GPUs** for small models.

---

## 5. Status of Engineering Phases

- **Phases 1-4**: Code written and verified to run (scaffolding for `NeuralOpsAttentionBackend`, `NeuralOpsRecurrentSpec`, `LinearAttnModelSpec`, and TensorRT-LLM plugin).
- **Phase 5**: Benchmark executed; results show vLLM is not faster than PyTorch on T4, SGLang lacks Turing binaries, and TensorRT-LLM requires a Python 3.12 environment.

---

## 6. TensorRT-LLM Environment Prerequisites

- **Python Version Constraint**: TensorRT-LLM requires Python 3.10 or 3.12. NVIDIA does not distribute precompiled wheels for Python 3.13 on PyPI.
- **Kaggle Environment**: The default environment on Kaggle runs Python 3.13.15 (as of November 2026), preventing direct `pip install tensorrt_llm`.
- **How to Test TRT-LLM**: Spin up a dedicated Docker container or virtual environment with **Python 3.12** and **CUDA 12.x / 13.x**:
  ```bash
  docker run --gpus all -it --rm nvcr.io/nvidia/tensorrt:24.05-py3
  pip install tensorrt_llm -U --pre --extra-index-url https://pypi.nvidia.com
  ```

---

## 7. Exact Reproduction Commands (Kaggle T4)

To reproduce the benchmark figures directly on the Kaggle remote host:

### 1. Host Inspection
```bash
ssh kaggle "nvidia-smi"
```

### 2. Run Single vLLM Benchmark
```bash
ssh kaggle "cd /kaggle/working/NeuralOps && python3 scripts/benchmark_serving.py --engine vllm --model gpt2 --attention cosformer --seq-len 512"
```

### 3. Run Comprehensive Cross-Engine Benchmark
```bash
ssh kaggle "cd /kaggle/working/NeuralOps && python3 scripts/benchmark_serving.py --engine all --model gpt2 --attention cosformer,gla_attention,decay_linear_attention --seq-len 512"
```

### 4. Run Integration Unit Test Suite
```bash
ssh kaggle "cd /kaggle/working/NeuralOps && pytest tests/integrations/test_tensorrt_llm_plugin.py tests/integrations/test_sglang_backend.py -v --tb=short"
```
