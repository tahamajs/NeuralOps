# Empirical Evaluation & Architectural Analysis Report: KVTC Compressor

**Author / Evaluator:** Antigravity AI  
**Date:** 2026-09-21  
**Target Module:** [`Codes/framework/compression_engine/kvtc_compressor.py`](Codes/framework/compression_engine/kvtc_compressor.py)  
**Test Suite:** [`Codes/framework/tests/test_kvtc_hardened.py`](Codes/framework/tests/test_kvtc_hardened.py)  

---

## 1. Executive Summary & Architectural Clarification

### 1.1 Weight Quantization vs. Dynamic KV-Cache Compression
- **Previous Framing / Misconception:** The module name "KVTC" previously suggested dynamic Key-Value Cache Token Compression during autoregressive generation.
- **Architectural Truth:** Inspection of the implementation demonstrates that this compressor operates exclusively as an **offline static weight-matrix compressor** combining **Principal Component Analysis (PCA) projection** with **integer scalar quantization** (with optional rank-1 LoRA residual recovery and theoretical entropy coding estimates).
- **Hardening & Alias Resolution:**
  - To prevent misleading scientific framing while preserving backward compatibility, the class has been clearly documented with `IS_DYNAMIC_KV_CACHE = False`.
  - Registered aliases in [`Codes/framework/compression_engine/kvtc_compressor.py`](Codes/framework/compression_engine/kvtc_compressor.py):
    - Original keys: `"kvtc"`, `"kvtc_quantizer"`, `"kvtc_factory"`
    - Honest aliases: `"pca_quant"`, `"pca_quant_compressor"`, `"pca_quant_factory"`

---

## 2. Hardening & Bug Fixes Implemented

1. **Signed Integer Quantization Storage (`torch.int8`):**
   - Fixed silent wrap-around caused by unsigned arithmetic in signed `torch.int8` buffers.
   - Changed quantization to symmetric signed integer bounds:
     $$\text{scale} = \frac{x_{\max}}{2^{b-1} - 1}, \quad q = \text{clamp}\left(\left\lfloor \frac{x}{\text{scale}} \right\rceil, -2^{b-1}, 2^{b-1}-1\right)$$
   - Ensured buffers store actual `torch.int8` values where `buffer.element_size() == 1`.
   - Reduced scalar reconstruction relative error from **2.89** to **0.0078** on unit tests.

2. **Percentile-Based Outlier Clipping:**
   - Integrated `_percentile_clip` (low=0.1%, high=99.9%) prior to scale computation to prevent activation/weight outlier spikes from destroying quantization precision.

3. **Accurate Physical Memory Byte Tracking:**
   - Replaced parameter length heuristics with exact pointer-deduplicated buffer and parameter byte accounting (`data_ptr()`), correctly tracking `int8` storage vs standard `fp16`/`fp32`.

4. **Forward Pass Verification:**
   - Injected `forward_check_success` into `compress_model` to verify numerical stability and shape validity on dummy inputs prior to returning the compressed model.

5. **GQA Asymmetry Protection:**
   - Added `skip_attention_qkv=True` by default, protecting grouped-query attention models (where key/value head counts differ from query head counts) from dimension mismatch crashes.

6. **Entropy Coding Grounding:**
   - Renamed metrics in `SimpleEntropyCoder` to include `entropy_lower_bound_bits` and explicitly documented in `kvtc_stats.json`:
     `"Theoretical lower bound; actual bitstream not implemented."`

7. **VRAM Protection (`--max_vram_fraction`):**
   - Added GPU memory ceiling parameter in [`run_quality_sweep.py`](Codes/framework/run_quality_sweep.py) via `torch.cuda.set_per_process_memory_fraction`, ensuring non-disruptive execution alongside the background daemon `auto_pipeline.py`.

---

## 3. Unit Test Verification

The hardened test suite [`Codes/framework/tests/test_kvtc_hardened.py`](Codes/framework/tests/test_kvtc_hardened.py) tests 8 critical invariants:

| Test Name | Description | Status |
|---|---|:---:|
| `test_weight_quantization_not_kv_cache` | Asserts `IS_DYNAMIC_KV_CACHE == False` and operates on linear weights | **PASS** |
| `test_integer_quantization` | Verifies `torch.int8` dtype, element size = 1, and reconstruction accuracy | **PASS** |
| `test_forward_check_success` | Verifies forward pass validation runs and reports `True` | **PASS** |
| `test_gqa_asymmetric_skipped` | Verifies asymmetric GQA layers are gracefully skipped | **PASS** |
| `test_percentile_clip` | Checks clipping reduces outlier distortion | **PASS** |
| `test_real_bytes_measurement` | Confirms deduplicated physical byte measurements | **PASS** |
| `test_registry_aliases` | Tests both `"kvtc"` and `"pca_quant"` registrations | **PASS** |
| `test_end_to_end_compression` | Runs compression + forward pass on miniature transformer | **PASS** |

---

## 4. Empirical Evaluation on DistilBERT / SST-2

### 4.1 8-bit PCA-Quant Benchmark Results (Multi-Seed)

Execution Settings:
- Model: `distilbert-base-uncased` (Classification on `sst2`)
- Batch size: 8, Precision: `fp16`, Parallel: 1, Recovery steps: 50
- Compression target: Attention linear projections (24 layers total)

| Metric | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|---|:---:|:---:|:---:|:---:|
| **Status** | **PASS** | **PASS** | **PASS** | **100% PASS** |
| **Accuracy (%)** | 67.43 | 68.12 | 67.09 | **67.55 ± 0.53** |
| **F1 Score** | 0.7526 | 0.7561 | 0.7507 | **0.7531 ± 0.0027** |
| **F1 Macro** | 0.6381 | 0.6479 | 0.6333 | **0.6398 ± 0.0074** |
| **Validation Loss** | 0.6031 | 0.6023 | 0.6050 | **0.6035 ± 0.0014** |
| **Peak VRAM (MB)** | 384.55 | 384.55 | 384.55 | **384.55 ± 0.00** |
| **Throughput (tok/s)**| 28,735.6 | 31,128.3 | 22,618.0 | **27,494.0 ± 4,388.9** |
| **Avg Recon Error** | 0.3194 | 0.3194 | 0.3194 | **0.3194 ± 0.00** |
| **Layers Compressed**| 24 / 24 | 24 / 24 | 24 / 24 | **24 / 24 (100%)** |
| **Fallbacks** | 0 | 0 | 0 | **0** |
| **Params (Before)** | 66,955,010 | 66,955,010 | 66,955,010 | 66.96M |
| **Params (After)** | 53,389,058 | 53,389,058 | 53,389,058 | **53.39M (-20.26%)** |

### 4.2 4-bit PCA-Quant Benchmark Results (Multi-Seed)

Execution Settings:
- Model: `distilbert-base-uncased` (Classification on `sst2`)
- Batch size: 4, Precision: `fp16`, Parallel: 1, Recovery steps: 50
- Compression target: Attention linear projections (24 layers total)
- Max VRAM fraction cap: 50% (`--max_vram_fraction 0.50`)

| Metric | Seed 42 | Seed 123 | Seed 456 | Mean ± Std |
|---|:---:|:---:|:---:|:---:|
| **Status** | **PASS** | **PASS** | **PASS** | **100% PASS** |
| **Accuracy (%)** | 66.51 | 65.60 | 65.37 | **65.83 ± 0.60** |
| **F1 Score** | 0.7470 | 0.7427 | 0.7414 | **0.7437 ± 0.0029** |
| **F1 Macro** | 0.6260 | 0.6118 | 0.6086 | **0.6155 ± 0.0093** |
| **Validation Loss** | 0.6026 | 0.6044 | 0.6051 | **0.6040 ± 0.0013** |
| **Peak VRAM (MB)** | 384.55 | 384.55 | 384.55 | **384.55 ± 0.00** |
| **Throughput (tok/s)**| 107,411.8 | 85,229.1 | 106,622.1 | **99,754.3 ± 12,586.2** |
| **Avg Recon Error** | 0.3485 | 0.3485 | 0.3485 | **0.3485 ± 0.00** |
| **Layers Compressed**| 24 / 24 | 24 / 24 | 24 / 24 | **24 / 24 (100%)** |
| **Fallbacks** | 0 | 0 | 0 | **0** |
| **Params (Before)** | 66,955,010 | 66,955,010 | 66,955,010 | 66.96M |
| **Params (After)** | 53,389,058 | 53,389,058 | 53,389,058 | **53.39M (-20.26%)** |

### 4.3 Compression & Memory Savings Breakdown
- **Physical Byte Footprint:**
  - Uncompressed attention bytes: 267,824,136 bytes (~255.4 MB)
  - Compressed model physical bytes: 227,716,296 bytes (~217.2 MB)
  - Physical memory reduction: **14.98%** (due to embedding and classifier layers remaining dense FP16)
- **Theoretical Entropy Bounds:**
  - Average entropy across weights: **7.19 bits / symbol** (8-bit)
  - Theoretical Shannon compression ratio: **2.23×**
- **Quantization Trade-off (8-bit vs 4-bit):**
  - Moving from 8-bit to 4-bit incurs only a minor **~1.72% drop in accuracy** (67.55% → 65.83%) while retaining full model stability across all seeds.

---

## 5. Recommendation for Downstream Use & Publications

1. **Terminology in Papers / Documentation:**
   Always describe KVTC as **PCA-Quant (Principal Component Analysis + Integer Quantization)**. Do not classify it as a dynamic KV-cache eviction or token-dropping policy.
2. **Registry Use:**
   Standardize future experiments on the alias `--modules pca_quant`, using `--modules kvtc` solely for legacy script compatibility.
3. **Execution Safety:**
   Always supply `--max_vram_fraction 0.5` on shared workstation or multi-job environments to preserve GPU headroom for the pipeline scheduler.
