# 🗺️ TensorRT-LLM Integration Technical Roadmap (v2.0.0)

## 1. Executive Summary & Architectural Vision
NeuralOps (`SVD_linear_Attention` v1.3.7) provides production-grade serving support for Hugging Face Transformers, ONNX Runtime graph export, and in-tree continuous batching adapters for vLLM, TGI, and SGLang. 

For maximum deployment density, minimal time-to-first-token (TTFT), and ultra-low latency in enterprise GPU clusters, **TensorRT-LLM** represents the flagship production runtime. TensorRT-LLM relies on specialized C++/CUDA plugins conforming to NVIDIA's `nvinfer1::IPluginV2DynamicExt` and `tensorrt_llm::plugins::api`.

This roadmap outlines the systematic engineering execution to bring native TensorRT-LLM compilation, custom kernel compilation, and optimized engine execution into **NeuralOps v2.0.0**.

---

## 2. Required Custom TensorRT C++ Plugins

The core mathematical primitives of NeuralOps must be compiled into native TensorRT plugin libraries (`.so`):

1. **`FusedAttentionPlugin`**:
   - Compiles subquadratic and low-rank linear attention mechanisms (such as SVD Linear Attention, Cosformer, and Gated Linear Attention).
   - Performs online recurrent state accumulation ($S_t = S_{t-1} + K_t^T V_t$) in FP16 and BF16 precision.
   - Eliminates standard transformer $O(T^2)$ KV-cache footprint in favour of constant $O(1)$ recurrent memory per sequence.

2. **`QuantizedGEMMPlugin`**:
   - Supports compressed low-rank and quantized projections, specifically W4A16 (INT4 packed weights with FP16 activations) and FP8 matrix multiplications.
   - Implements fast bit-shift unpack kernels and fused dequantization-GEMM on NVIDIA Tensor Cores.

3. **`SelectiveScanPlugin`**:
   - Native TensorRT dynamic shape implementation for state-space models (Mamba/SSM) and associative scans.
   - Supports variable sequence lengths and continuous batching chunk prefill.

4. **`PagedAttentionPlugin`**:
   - Paged recurrent state serving allocating fixed-size GPU virtual blocks for recurrent states ($H \times D \times D$) or sparse KV slots.
   - Mimics BlockTable memory layouts to enable zero-copy sequence forking and dynamic request admission.

---

## 3. Build System & Compilation Infrastructure

To compile these plugins, NeuralOps v2.0.0 will introduce dedicated CMake and PyTorch C++ extension build hooks:

- **Build System Changes**:
  - `setup.py` custom build extension flag `--cuda-ext` passing `nvcc` flags and linking against TensorRT libraries.
  - Automatic discovery of TensorRT headers via `$TENSORRT_DIR/include` and library paths via `$TENSORRT_DIR/lib`.
- **Directory Layout**:
  ```
  csrc/
  ├── CMakeLists.txt
  ├── plugins/
  │   ├── fused_attention/
  │   │   ├── fused_attention_plugin.h
  │   │   ├── fused_attention_plugin.cu
  │   │   └── fused_attention_kernel.cuh
  │   ├── quantized_gemm/
  │   │   ├── quantized_gemm_plugin.h
  │   │   └── quantized_gemm_plugin.cu
  │   ├── selective_scan/
  │   │   ├── selective_scan_plugin.h
  │   │   └── selective_scan_plugin.cu
  │   └── paged_attention/
  │       ├── paged_attention_plugin.h
  │       └── paged_attention_plugin.cu
  └── tensorrt_llm_bindings.cpp
  ```
- **Build Configurations**:
  - `CUDA_ARCHITECTURES`: `80;86;89;90` (Ampere, Ada Lovelace, Hopper).
  - Target NVIDIA TensorRT SDK: `10.0+`.
  - Linkage against `libnvinfer.so`, `libnvinfer_plugin.so`, and `libtensorrt_llm.so`.

---

## 4. Dependencies and Environment Constraints

- **Hardware**: NVIDIA GPU with Compute Capability $\ge 8.0$ (Ampere RTX 30xx/A100, Ada Lovelace RTX 40xx/L40, Hopper H100).
- **CUDA Toolkit**: Version $\ge 12.2$.
- **TensorRT SDK**: Version $10.0$ or higher with C++ headers (`NvInfer.h`, `NvInferPlugin.h`).
- **cuDNN**: Version $9.0$ or higher.
- **Operating System Constraints & Blocking Items**:
  - TensorRT-LLM and custom CUDA plugins are strictly supported on **Linux x86_64 and Windows WSL2**.
  - **No macOS Support**: Apple Silicon workstations cannot build or execute TensorRT-LLM plugins due to the lack of NVIDIA GPU drivers and NVCC toolchains. On macOS, NeuralOps defaults to native PyTorch and MPS accelerated backends.

---

## 5. Verification & Testing Strategy

1. **Unit Plugin Verification**:
   - Compare outputs of compiled TensorRT custom plugins against existing Triton JIT reference kernels and PyTorch eager implementations across random synthetic inputs (strict tolerance $\epsilon \le 10^{-4}$ in FP16).
2. **Runtime Smoke Testing**:
   - Compile plugin into shared library, load into TensorRT runtime engine, and execute a smoke test verifying end-to-end token generation on GPT-2 and RoBERTa checkpoints.
3. **End-to-End Engine Latency Benchmark**:
   - Benchmark throughput (tokens/sec) and time-to-first-token (TTFT) on NVIDIA RTX 4070 (SM 8.9) and A100 (SM 8.0) across sequence lengths $L \in \{2048, 8192, 32768, 131072\}$.
4. **Speculative Decoding Alignment**:
   - Pair compiled TensorRT draft engines with target models using NeuralOps `SelfSpeculativeDecoder`.

---

## 6. Implementation Milestones & Timeline

| Milestone | Deliverables |
| :--- | :--- |
| **M1: C++ Plugin Scaffold** | Plugin classes implementing `IPluginV2DynamicExt`, serialized state methods, and parameter parsing. |
| **M2: CUDA Kernel Ports** | Port existing 26 Triton JIT kernels into optimized standalone CUDA/C++ templates. |
| **M3: Model Engine Builder** | Python `TensorRTLLMPluginBuilder.export_engine()` assembling complete model networks via TRT NetworkDefinition. |
| **M4: Production v2.0.0 Release** | Automated wheel packaging with pre-compiled `.so` plugins for Linux x86_64 and CUDA 12.x. |

---

## 7. Official References & Standards
- [NVIDIA TensorRT-LLM Architecture Guide & Repository](https://github.com/NVIDIA/TensorRT-LLM)
- [TensorRT Custom Plugin Developer Guide (`IPluginV2DynamicExt`)](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [TensorRT-LLM C++ Plugin Architecture and Custom Layers Documentation](https://nvidia.github.io/TensorRT-LLM/advanced/custom-plugins.html)
