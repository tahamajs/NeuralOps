# NeuralOps Technical Roadmap

This roadmap outlines the major architectural milestones and development objectives for **NeuralOps** — the unified open-source framework for subquadratic attention and deep model compression.

---

## 🗺️ Version Roadmap

### ✅ v1.0 — Foundation & Unification (June 2026)
- [x] Initial public release on PyPI.
- [x] Core drop-in attention replacement engine (`neuralops.replace_attention`).
- [x] SVD and low-rank compression pipeline (`neuralops.compress`).
- [x] Zero-shot attention surgery across 8 transformer architectures.

### ✅ v1.1 — Triton Acceleration & Benchmarking (September 2026)
- [x] First fused Triton kernels: `fused_rope`, `gdn_scan`, `gla_scan`, `hgrn2_scan`, `fms_scan`.
- [x] Knowledge Distillation (KD) teacher-student training engine.
- [x] Multi-dataset benchmark sweep infrastructure.

### ✅ v1.2 — Unified CLI & Registry Polish (October 2026)
- [x] Single entrypoint CLI (`neuralops run`, `plan`, `list`, `verify`, `resume`, `results`).
- [x] Formalized 131 canonical attention mechanisms and 72 model compression engines.
- [x] Deprecated 657 legacy aliases with backward-compatible dynamic redirection.
- [x] Full root directory consolidation.

### 🚀 v1.3 — Production MLOps & Compilation (Current Release - October 2026)
- [x] **`torch.compile` Compatibility**: Zero graph-break custom op wrappers and Meta/FakeTensor shape registration (`neuralops/core/triton_kernels/op_registry.py`).
- [x] **Production Export Subsystem**: `neuralops export` generating Safetensors with `neuralops_config.json`, deterministic CPU ONNX, and TorchScript.
- [x] **Enterprise MLOps**: Integrated Weights & Biases, MLflow, and TensorBoard experiment tracking.
- [x] **Serving Adapters**: Native hooks for vLLM paged KV cache tables and TGI container runtimes.
- [x] **Parity Test Suite**: Strict numerical parity verification between Triton kernels and PyTorch fallbacks across FP32, FP16, and BF16.

### 🔮 v1.4 — Native FP8 / INT4 Quantization & Blackwell Architecture (Q1 2027)
- [ ] Native FP8 (E4M3 / E5M2) GEMM and attention kernels for NVIDIA Hopper & Blackwell (H100, H200, B200).
- [ ] 4-bit block-wise quantization with Triton dequantization fusion.
- [ ] Native AMD ROCm CDNA3 / ROCm 6.x Triton kernel tuning and autotuner matrix cache.

### 🔮 v1.5 — Native vLLM Plugin & GGUF Quantization Export (Q2 2027)
- [ ] First-class vLLM plugin package (`vllm-plugin-neuralops`) enabling instant serving of any of the 131 subquadratic attention models in high-throughput clusters.
- [ ] Direct export to GGUF format for llama.cpp, Ollama, and local edge deployment.
- [ ] AWQ and GPTQ co-compression support alongside SVD rank reduction.

### 🔮 v2.0 — Distributed Sequence Parallelism & Ultra-Long Context (Q3-Q4 2027)
- [ ] DeepSpeed Ulysses and Ring Attention sequence parallelism for 1M+ token context lengths.
- [ ] Unified hybrid attention compiler (automatically discovering optimal layer-wise mixtures of linear, sliding window, and full attention).
- [ ] Fully asynchronous, pipeline-parallel knowledge distillation across multi-node clusters.
