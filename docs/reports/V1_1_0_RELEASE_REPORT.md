# NeuralOps v1.1.0 Release Report

**Release Date:** October 8, 2026  
**Version:** `1.1.0`  
**Package Distributions:**
- `neuralops-1.1.0-py3-none-any.whl` (1.9 MB)
- `neuralops-1.1.0.tar.gz` (1.6 MB)
**TestPyPI URL:** [https://test.pypi.org/project/neuralops/1.1.0/](https://test.pypi.org/project/neuralops/1.1.0/)

---

## 1. Executive Summary

NeuralOps v1.1.0 represents a milestone upgrade, transitioning preliminary skeletons into production-grade, mathematically verified, high-performance attention and compression engines.

### Key Milestones Delivered:
1. **Production Quantization & Pruning**:
   - Complete GPTQ (Hessian inverse, Cholesky decomposition, block updates).
   - AWQ (activation-aware grid search for per-channel scaling).
   - SmoothQuant (mathematical migration of activation outliers into weights).
   - Wanda (magnitude $\times$ input norm pruning).
   - LoRA / QLoRA / DoRA weight merging and extraction.
2. **Serving & Decoding Stack**:
   - Production inference engine with speculative decoding (draft-target verification) and continuous batching.
   - Beam search decoder with length penalty, repetition penalty, and n-gram blocking.
   - Mixed-precision training runner supporting gradient accumulation, gradient clipping, and AMP.
3. **Triton Kernels with Transparent Fallbacks**:
   - Sliding window attention, Paged attention (block lookup tables), Flash-decoding split-KV.
   - Fused RoPE (rotary positional embeddings).
   - Fused LayerNorm/RMSNorm + Attention.
   - FP8/INT8 quantized matrix multiplication kernels.
   - Fused elementwise activations (SwiGLU, GELU, RMSNorm).
   - 100% CPU/non-CUDA transparent fallbacks.
4. **Rigorous Numerical Verification & Benchmarking**:
   - Autograd `gradcheck` tests on attention mechanisms.
   - Edge case validation (zero-length padding, arbitrary aspect ratios, inf/nan handling).
   - Comprehensive benchmark suite (`suite_runner.py`) for speed, peak memory, and sequence length scaling sweeps.
5. **Ecosystem Integrations & Hardening**:
   - Zero-dependency HuggingFace attention patching (`patch_model_attention`).
   - Adapters for vLLM, DeepSpeed, and PyTorch FSDP.
   - Strict typing, logging audit, input validation, and registry invariant verification.

---

## 2. Registry & Artifact Verification

- **Attention Registry Count**: `145` canonical modules (Preserved $\ge 145$).
- **Compressor Registry Count**: `73` canonical modules (Preserved $\ge 73$).
- **Deprecated Aliases**: `658+` aliases preserved in `neuralops/core/deprecated_aliases.json`.
- **Wheel Archive Size**: `1.9 MB` (Target: $< 5.0\text{ MB}$, zero raw `.arrow`/`.parquet`/`.h5` files packaged).
- **Twine Validation**: `PASSED` (`twine check dist/*`).
- **TestPyPI Distribution**: Successfully uploaded and live at [https://test.pypi.org/project/neuralops/1.1.0/](https://test.pypi.org/project/neuralops/1.1.0/).

---

## 3. Git Commit History for Release

| Phase | Commit | Description |
|---|---|---|
| **Phase 1** | `859263175d` | feat: production-grade GPTQ quantization |
| **Phase 1** | `2676199098` | feat: production-grade AWQ quantization |
| **Phase 1** | `f961f3fd76` | feat: production-grade SmoothQuant |
| **Phase 1** | `26cf7c4691` | feat: production-grade Wanda pruning |
| **Phase 1** | `5f6fda1f84` | feat: LoRA merge/extraction, QLoRA, and DoRA |
| **Phase 1** | `b97fefad61` | feat: speculative decoding, beam search, continuous batcher |
| **Phase 1** | `70f92cdcd1` | feat: MixedPrecisionTrainer with grad accum & clipping |
| **Phase 2** | `ed50b2d210` | feat(kernels): sliding window attention Triton kernel + fallback |
| **Phase 2** | `51525718f7` | feat(kernels): paged attention Triton kernel + fallback |
| **Phase 2** | `3c22719b8a` | feat(kernels): Flash-decoding split-KV Triton kernel + fallback |
| **Phase 2** | `a229d26d49` | feat(kernels): fused RoPE Triton kernel + fallback |
| **Phase 2** | `28bf9d01b7` | feat(kernels): fused norm+attention Triton kernel + fallback |
| **Phase 2** | `cd4e41646d` | feat(kernels): FP8/INT8 and elementwise fused kernels + fallbacks |
| **Phase 3** | `38ecdb133e` | test: numerical correctness and autograd gradcheck suite |
| **Phase 3** | `dec42454ef` | test: comprehensive edge cases (seqlen 1, large seqlen, inf/nan) |
| **Phase 3** | `5b61ed9147` | test: integration pipelines and regression benchmarks |
| **Phase 4** | `8df27656ba` | perf(benchmarks): comprehensive benchmark suite (speed, mem, sweep) |
| **Phase 5** | `70053bb9af` | docs: complete documentation suite (API reference, tutorials, architecture) |
| **Phase 6** | `4675af5eef` | feat(integrations): complete all integrations (HF patch, vLLM, DeepSpeed, FSDP) |
| **Phase 7** | `017e7eceef` | refactor: production hardening, logging audit, and type validation |
| **Phase 8** | `a889b378c5` | feat: release version 1.1.0 with new features and enhancements |
