# Changelog

All notable changes to `NeuralOps` (`SVD_linear_Attention`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.3.7] - 2026-10-16

### Added
- **Production Serving Integrations**:
  - `neuralops/integrations/tgi.py`: Full Hugging Face Text Generation Inference (TGI) continuous batching adapter, per-request stateful token streaming (`TGIStreamingIterator`), and FastAPI integration hook.
  - `neuralops/integrations/sglang.py`: Full SGLang RadixAttention prefix caching adapter (`SGLangRadixAdapter`) with node splitting, token chunking, and thread-safe LRU eviction.
  - `neuralops/integrations/tensorrt_llm.py`: Complete `TensorRTLLMPluginBuilder`, dynamic profile builder, C++ CUDA plugin source generator (`svd_linear_attention_plugin.cu/h`), and `TensorRTRuntimeRunner`.
  - Comprehensive 21-test serving test suite (`tests/test_framework_serving_adapters.py`).
- **Comprehensive Documentation**:
  - Full Material for MkDocs technical documentation suite covering API references (131 Attentions, 72 Compressors, 26 Triton Kernels, 6 Serving Frameworks), interactive tutorials, and benchmarks.
  - Comprehensive TensorRT-LLM roadmap specification (`docs/roadmap/TENSORRT_LLM_PLAN.md`).
- **Hugging Face Model Zoo Cards**:
  - Production model cards for `gpt2-medium-fwsvd-rank64`, `roberta-base-fwsvd-rank32`, and `distilbert-base-fwsvd-rank64`.
- **CI/CD Automation**:
  - Multi-tier GitHub Actions workflows for continuous integration testing, documentation deployment, and PyPI release packaging.

### Changed
- Reconciled all README framework integration paths with the verified codebase structure.
- Hardened Gated DeltaNet Triton JIT scan (`neuralops/core/triton_kernels/gdn_scan.py`) with support for sequence masks and initial recurrent states.
- Cleaned up internal deprecation paths from `neuralops.core.compression_engine` to `neuralops.core.compression_pipeline`.

---

## [1.3.6] - 2026-09-30

### Added
- Multi-pillar KD recovery loop with progressive layer-wise compression.
- AdaSVD and FWSVD adaptive rank allocation routines.
- Expanded Triton JIT kernel library to 26 canonical kernels with unified auto-tuning.

---

## [1.3.0] - 2026-08-15

### Added
- Hugging Face `replace_attention()` dynamic injector and AutoModel roundtrip serialization.
- ONNX Runtime graph export support with custom linear attention operator bindings.
- PagedStateTable KV cache manager for vLLM block paging.
