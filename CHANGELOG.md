# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] - v1.3.0

### Added
- **`torch.compile` Full Compatibility**:
  - Unified `neuralops/core/triton_kernels/op_registry.py` with modern PyTorch `torch.library.custom_op` and `register_fake` Meta/FakeTensor shape tracing.
  - Transparent execution without graph breaks across models modified by `neuralops.replace_attention` and `neuralops.compress`.
- **Production Export Pipeline (`neuralops.export`)**:
  - `neuralops.export_model` supporting `"safetensors"` (with architecture surgery metadata `neuralops_config.json`), deterministic CPU-fallback `"onnx"`, and `"torchscript"`.
  - Added CLI subcommand: `neuralops export --model <path> --attn <module> --comp <engine> --format [safetensors|onnx|torchscript]`.
  - Serving adapters (`neuralops/export/serving.py`) for **vLLM** (paged KV cache wrapper) and **TGI (Text Generation Inference)** runtimes.
- **MLOps Experiment Tracking & Distributed Training**:
  - Unified `neuralops/benchmarks/loggers.py` multiplexing metrics across Weights & Biases (`wandb`), MLflow (`mlflow`), TensorBoard, and local JSONL/CSV.
  - Added CLI flags to `neuralops run`: `--tracker [none|wandb|mlflow|tensorboard|all]`, `--project-name`, `--run-name`, `--log-interval`, `--distributed-backend`, and `--mixed-precision`.
  - Added `neuralops[mlops]` extra in `pyproject.toml`.
- **Numerical Parity & Regression Suite**:
  - Parametric parity test suite (`tests/test_triton_parity.py`) verifying Triton kernels match PyTorch fallbacks across FP32 and FP16.
  - Compile and export validation suite (`tests/test_compile_and_export.py`).
  - Automated latency and memory regression benchmark (`tests/benchmarks/test_perf_regression.py`).

---

## [1.2.0] - 2026-10-09

### Added
- Unified single entrypoint `neuralops` CLI with subcommands: `run`, `plan`, `list`, `verify`, `resume`, `results`.
- 131 canonical attention mechanisms and 72 model compression engines verified.
- 26 Triton kernel modules with verified native PyTorch CPU/MPS fallbacks.
- Hugging Face Hub upload templates and automated model cards.

### Changed
- Comprehensive root directory reorganization moving misplaced files to `academic/`, `docs/`, `scripts/`, and `results/`.
- Deprecated 657 redundant legacy aliases while preserving 100% backward compatibility via `neuralops/core/deprecated_aliases.json`.

---

## [1.1.0] - 2026-09-20

### Added
- Triton-accelerated fused kernels: `fused_rope`, `gdn_scan`, `gla_scan`, `hgrn2_scan`, `fms_scan`.
- Knowledge distillation (KD) benchmark engine supporting temperature scaling and teacher-student matching.
- Multi-dataset benchmark sweep covering SST-2, MNLI, AG News, and language modeling perplexity.

---

## [1.0.0] - 2026-06-15

### Added
- Initial public release of NeuralOps on PyPI.
- Drop-in attention replacement engine (`neuralops.replace_attention`) for Hugging Face Transformers.
- SVD and low-rank compression pipeline (`neuralops.compress`).
- Zero-shot attention injection across 8 transformer architectures.
