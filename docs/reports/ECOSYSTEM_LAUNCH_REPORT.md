# NeuralOps Ecosystem Launch Report

## Version: 1.2.0
## Date: 2026-10-09
## Release Lead: Taha Majlesi

---

## 1. PyPI Distribution Status

- **Status**: Live on PyPI
- **Version**: 1.2.0
- **PyPI URL**: https://pypi.org/project/neuralops/1.2.0/
- **Distribution Artifacts**:
  - `neuralops-1.2.0-py3-none-any.whl` (1.9 MB)
  - `neuralops-1.2.0.tar.gz` (1.6 MB)
- **Dependencies**: Minimal pure-Python core (`packaging>=23.0`) with 9 modular extras (`cuda`, `rocm`, `mps`, `cpu`, `eval`, `benchmark`, `dev`, `docs`, `all`).

---

## 2. Hugging Face Hub Integration

- **Model Card Template & Bundle**: Pre-packaged under `hf_upload/distilbert-svd-attention/README.md`
- **Target Repository**: `tahamajs/distilbert-svd-attention`
- **Upload Guide**: Detailed step-by-step workflow documented in `docs/HUGGINGFACE_UPLOAD.md`
- **Hub Variants Configured**:
  - `neuralops-distilbert-sst2-r32`
  - `neuralops-distilbert-sst2-r64`
  - `neuralops-distilbert-sst2-r128`
  - `neuralops-roberta-sst2-r64`

---

## 3. GitHub & Community Infrastructure

- **Repository**: https://github.com/tahamajs/SVD_linear_Attention
- **Branding Assets**:
  - Vector social preview banner: `assets/social_preview.svg` (1280x640)
  - Vector header banner: `assets/banner.svg` (1200x300)
- **Community Standards**:
  - Issue templates: Bug Report, Feature Request, Documentation Update
  - Pull Request template: `.github/PULL_REQUEST_TEMPLATE.md`
  - Security Policy: `SECURITY.md`
  - Sponsorship & Funding: `.github/FUNDING.yml`
- **Repository Guidelines**: Documented in `docs/GITHUB_SETTINGS.md`

---

## 4. Documentation Hub

- **Root [README.md](file:///Users/tahamajs/Documents/uni/Project/README.md)**: Fully refreshed with v1.2.0 announcement, platform matrix, 159 attention mechanisms, 73 compressors, 19 Triton kernels, benchmarks, and citations.
- **Documentation Portal [docs/README.md](file:///Users/tahamajs/Documents/uni/Project/docs/README.md)**: Complete guide directory.
- **Per-Family Module Docs**: 18 family markdown files generated under `docs/MODULES/`.
- **v2.0 Roadmap**: Comprehensive evolution plan through 2027 outlined in `docs/ROADMAP.md`.

---

## 5. Verification Matrix

| Invariant / Check | Target | Actual | Status |
|:---|:---:|:---:|:---:|
| Attention Registry Count | 159 | 159 | ✓ PASS |
| Compressor Registry Count | 73 | 73 | ✓ PASS |
| Transitive Alias Resolution | Supported | Supported | ✓ PASS |
| Core Test Suite | 33 passed | 33 passed | ✓ PASS |
| Wheel Package Size | < 3.0 MB | 1.9 MB | ✓ PASS |
| PyPI Live Metadata | v1.2.0 | v1.2.0 | ✓ PASS |
| Git Branch Sync | Synchronized | Synchronized | ✓ PASS |

---

## 6. Next Milestones

- **v1.3.0 (Q1 2027)**: Intel XPU, Apple Metal (MPS), and Gaudi hardware kernels.
- **v1.4.0 (Q2 2027)**: Distributed context parallelism (>128k sequence length) and streaming generation runtime.
- **v1.5.0 (Q3 2027)**: ONNX, TensorRT-LLM, and CoreML export pathways.
- **v2.0.0 (Q4 2027)**: Sunset of legacy aliases and strictly typed API refactoring.
