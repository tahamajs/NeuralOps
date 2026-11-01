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

| Engine | Integration Status | Notes / Failure Diagnosis |
|---|---|---|
| **PyTorch (Baseline)** | **PASS** | Native HF model surgery using `replace_attention()`. Validated across `cosformer`, `gla_attention`, and `decay_linear_attention`. |
| **vLLM** | **PASS** | Custom backend scaffold (`NeuralOpsAttentionBackend`) and recurrent KV cache spec (`NeuralOpsRecurrentSpec`) integrated. End-to-end generation operational under eager mode. |
| **SGLang** | **FAIL** | Model registration (`LinearAttnModelSpec` with `uses_mamba_radix_cache=True`) and backend registry passed unit tests. Offline runtime engine failed due to `sgl_kernel` common_ops binary missing SM75 Turing support (`sm100/common_ops.abi3.so` PyTorch ABI symbol mismatch). |
| **TensorRT-LLM** | **DEFERRED** | Decorator plugin (`@python_plugin("NeuralOpsAttention")`) implemented in `neuralops.integrations.tensorrt_llm_plugin` with full offline unit test coverage (4/4 passed). Engine execution deferred due to lack of Python 3.13 Linux wheels from NVIDIA. |

---

## 3. Serving Benchmark Results

Empirical cold/warm-cache benchmarks conducted using `scripts/benchmark_serving.py` with `torch.cuda.Event` high-resolution hardware timers on model `gpt2` at sequence length 512.

```
===============================================================================================
                      NEURALOPS SERVING BENCHMARK RESULTS
===============================================================================================
| Engine         | Attention        | Seq Len   | Latency (ms)   | Throughput (tok/s)   | Peak VRAM (MB)   | Status     |
|----------------|------------------|-----------|----------------|----------------------|------------------|------------|
| pytorch        | cosformer        | 512       | 124.39         | 4116.19              | 617.04           | PASS       |
| pytorch        | gla_attention    | 512       | 90.08          | 5683.58              | 1191.25          | PASS       |
| pytorch        | decay_linear_attention | 512 | 46.22          | 11078.12             | 1195.60          | PASS       |
| vllm           | cosformer        | 512       | 209.09         | 2525.24              | 106.91           | PASS       |
| vllm           | gla_attention    | 512       | 214.13         | 2465.82              | 106.91           | PASS       |
| vllm           | decay_linear_attention | 512 | 220.59         | 2393.63              | 106.91           | PASS       |
| sglang         | cosformer        | 512       | N/A            | N/A                  | N/A              | FAIL       |
| sglang         | gla_attention    | 512       | N/A            | N/A                  | N/A              | FAIL       |
| sglang         | decay_linear_attention | 512 | N/A            | N/A                  | N/A              | FAIL       |
| tensorrt_llm   | cosformer        | 512       | N/A            | N/A                  | N/A              | DEFERRED   |
| tensorrt_llm   | gla_attention    | 512       | N/A            | N/A                  | N/A              | DEFERRED   |
| tensorrt_llm   | decay_linear_attention | 512 | N/A            | N/A                  | N/A              | DEFERRED   |
===============================================================================================
```

### Analysis
- **PyTorch Subquadratic Scaling**: `decay_linear_attention` achieved the highest raw throughput at **11,078 tokens/sec** with an average latency of **46.22 ms**, compared to 90.08 ms for `gla_attention` and 124.39 ms for `cosformer`.
- **vLLM Continuous Batching Overhead vs. Robustness**: vLLM maintained steady generation latencies of ~209–220 ms across all three recurrent mechanisms with low host process activation overhead (106.91 MB).

---

## 4. Commands to Reproduce on Kaggle

All benchmarks can be reproduced by SSHing to the remote Kaggle host and running:

### 1. Verify GPUs
```bash
nvidia-smi
```

### 2. Run Single vLLM Benchmark
```bash
cd /kaggle/working/NeuralOps
python3 scripts/benchmark_serving.py --engine vllm --model gpt2 --attention cosformer --seq-len 512
```

### 3. Run Comprehensive Cross-Engine Benchmark
```bash
cd /kaggle/working/NeuralOps
python3 scripts/benchmark_serving.py --engine all --model gpt2 --attention cosformer,gla_attention,decay_linear_attention --seq-len 512
```

### 4. Run Integration Test Suite
```bash
cd /kaggle/working/NeuralOps
pytest tests/integrations/test_tensorrt_llm_plugin.py tests/integrations/test_sglang_backend.py -v --tb=short
```
