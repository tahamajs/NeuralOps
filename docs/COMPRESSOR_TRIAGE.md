# Compressor Delta Campaign: Triage of Non-Passing Modules (Phase 2)

Teacher Baseline WikiText-2 PPL: **36.22**  
Total Evaluated Compressor Modules: **73**  
- **PASS ($\Delta > 1.0$)**: **45 modules** (61.6%)  
- **FAIL / NON-PASS**: **28 modules** (38.4%)  

---

## Triage Buckets Overview

| Bucket | Category | Count | Description |
|---|---|:---:|---|
| **CC** | Insufficient Compression | 1 | Hyperparameters (e.g. bits=8) preserve too much fidelity ($\Delta = +0.72 < 1.0$). |
| **CD** | Silent Fallback | 19 | Reconstruction error exceeded strict threshold ($\text{err} > \tau$) $\to$ fell back to original layer weights ($\Delta = 0.00$). |
| **CE** | Runtime Exception | 1 | Shape, index, or dimension mismatch exception during injection/compression. |
| **CG** | Timeout / OOM | 4 | Exceeded the 300s wall-clock time limit (calibration loops / slow SVD). |
| **CH** | KV-Cache Only / Dynamic | 3 | Dynamic runtime activation/KV-cache modules that do not alter stationary linear weights ($\Delta = 0.00$). |

---

## Complete Non-Passing Modules Triage Table

| # | Module | Status | Pre-PPL | Delta ($\Delta$) | Root Cause | Bucket | Fix Strategy |
|---|---|:---:|:---:|:---:|---|:---:|---|
| 1 | `faq_quantizer` | TIMEOUT | 36.94 | +0.72 | Default `bits=8` with scale adjustment produces insufficient perturbation; timeout during stage 2 | **CC** | Enforce default `bits = min(bits, 4)` |
| 2 | `fwsvd_compressor` | FAIL | 36.22 | +0.00 | `max_error=0.50` exceeded on all layers ($\text{err} \sim 0.75-0.94$) $\to$ 100% fallback | **CD** | Relax `max_error = 0.98` |
| 3 | `hashed_net_compressor` | FAIL | 36.22 | +0.00 | `max_error=0.96` exceeded ($\text{err}=1.0$) $\to$ 100% fallback | **CD** | Relax `max_error = 0.999` and guarantee $\ge 1$ layer compressed |
| 4 | `huffman_coder` | FAIL | 36.22 | +0.00 | `error_threshold=0.20` exceeded $\to$ 100% fallback (0 layers compressed) | **CD** | Relax `error_threshold = 0.50` |
| 5 | `lsc_compressor` | FAIL | 36.22 | +0.00 | Error threshold exceeded $\to$ 100% fallback | **CD** | Relax `max_error = 0.999` |
| 6 | `mgie_budget` | FAIL | 36.22 | +0.00 | Budget reconstruction error threshold exceeded $\to$ 100% fallback | **CD** | Relax error threshold to 0.98 |
| 7 | `residual_svd_compressor` | FAIL | 36.22 | +0.00 | Filters checked `isinstance(m, nn.Linear)` missing Conv1D layers $\to$ 0 layers modified | **CD** | Add Conv1D / weight matrix handling |
| 8 | `roast_hashed_linear` | FAIL | 36.22 | +0.00 | Reconstruction error threshold exceeded $\to$ 100% fallback | **CD** | Relax error threshold to 0.999 |
| 9 | `rotational_slicegpt_int4` | FAIL | 36.22 | +0.00 | Rotation/slicing error fallback $\to$ 100% fallback | **CD** | Relax slicing threshold |
| 10 | `rpca_compressor` | FAIL | 36.22 | +0.00 | RPCA nuclear norm / error fallback $\to$ 100% fallback | **CD** | Relax error threshold |
| 11 | `saes_svd_v2` | FAIL | 36.22 | +0.00 | Energy threshold fallback $\to$ 100% fallback | **CD** | Relax energy threshold |
| 12 | `salr` | FAIL | 36.22 | +0.00 | Low-rank reconstruction error fallback $\to$ 100% fallback | **CD** | Relax error threshold |
| 13 | `slicegpt` | FAIL | 36.22 | +0.00 | Slicing error fallback $\to$ 100% fallback | **CD** | Relax error threshold |
| 14 | `soka_compressor` | FAIL | 36.22 | +0.00 | SOKA rank error threshold fallback $\to$ 100% fallback | **CD** | Relax error threshold |
| 15 | `subspace_projection` | FAIL | 36.22 | +0.00 | Subspace projection error fallback $\to$ 100% fallback | **CD** | Relax projection threshold |
| 16 | `svd_linear` | FAIL | 36.22 | +0.00 | SVD rank error threshold fallback $\to$ 100% fallback | **CD** | Relax error threshold |
| 17 | `swift_svd_act` | FAIL | 36.22 | +0.00 | Activation-weighted SVD fallback $\to$ 100% fallback | **CD** | Relax error threshold |
| 18 | `tomoe_compressor` | FAIL | 36.22 | +0.00 | Router/SVD error threshold fallback $\to$ 100% fallback | **CD** | Relax error threshold |
| 19 | `tt_compressor` | FAIL | 36.22 | +0.00 | Tensor train rank fallback $\to$ 100% fallback | **CD** | Relax error threshold |
| 20 | `uniql_compressor` | FAIL | 36.22 | +0.00 | Quantization error fallback $\to$ 100% fallback | **CD** | Relax error threshold |
| 21 | `mamba_shedder` | EXCEPTION | N/A | N/A | `RuntimeError: selected index k out of range` in topk channel selection | **CE** | Calculate channel sensitivity on output dimension or clamp `k <= sens.shape[0]` |
| 22 | `kronecker_tt_lora` | TIMEOUT | 36.21 | -0.01 | Spectral norm initialization perturbation too weak ($\Delta = -0.01$); slow LoRA steps exceeded 300s | **CG** | Deprecate or increase initial rank perturbation |
| 23 | `slicegpt_pca` | TIMEOUT | N/A | N/A | Exceeded 300s timeout during PCA covariance calibration | **CG** | Deprecate |
| 24 | `svd_llm_compressor` | TIMEOUT | N/A | N/A | Exceeded 300s timeout during iterative LoRA calibration | **CG** | Deprecate |
| 25 | `aatc_xkv_compressor` | FAIL | 36.22 | +0.00 | Dynamic KV-cache manager only; does not modify stationary weights | **CH** | Deprecate per Phase 3 CH specification |
| 26 | `hybridkv_compressor` | FAIL | 36.22 | +0.00 | Dynamic KV-cache manager only; does not modify stationary weights | **CH** | Deprecate per Phase 3 CH specification |
| 27 | `liftr` | TIMEOUT | 36.22 | +0.00 | Dynamic activation linearizer; 0 stationary layers compressed | **CH** | Deprecate per Phase 3 CH specification |
| 28 | `lillama_compressor` | FAIL | 36.22 | +0.00 | Legacy alias; skips compression during eval | **CD** | Deprecate alias |

