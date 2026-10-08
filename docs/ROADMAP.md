# NeuralOps Roadmap

## Current: v1.2.0 (October 2026)

- 159 attention mechanisms across 11 families
- 73 compression engines across 7 families
- 19 Triton kernels with universal CPU & MPS fallbacks
- 8 supported transformer architectures for zero-shot attention injection
- Unified benchmark CLI runtime

## v1.3.0 (Q1 2027) — Hardware Expansion

### Planned
- Intel XPU / SYCL backend support
- Apple MPS customized Metal kernels
- Intel Gaudi accelerator support
- Huawei Ascend NPU backend
- FP8 native training integration

### Breaking Changes
- None

## v1.4.0 (Q2 2027) — Large Scale Distributed Optimization

### Planned
- PyTorch FSDP + Megatron Tensor Parallel integration
- Context parallel for sequences exceeding 128k tokens
- Streaming autoregressive generation runtime
- Dynamic speculative decoding with KV-cache recycling

### Breaking Changes
- None

## v1.5.0 (Q3 2027) — Serving & Edge Ecosystem

### Planned
- WebAssembly (Wasm) runtime for browser inference
- ONNX export with custom subquadratic operators
- TensorRT-LLM plugin integration
- CoreML optimization recipes for Apple devices
- Quantization-aware training (QAT) pipelines

### Breaking Changes
- None

## v2.0.0 (Q4 2027) — Major Architecture Refactor

### Planned
- Complete deprecation removal of the 657 legacy alias mappings
- Unified configuration parser
- Streamlined public API with typed functional factories
- Strict type-safe module registry
- Native asynchronous benchmarking runners

### Breaking Changes
- Removal of deprecated aliases
- Updated constructor argument schemas

---

## Long-term Vision

NeuralOps aims to serve as the foundational research toolkit for subquadratic attention mechanisms and transformer model compression across research and production deployments.
