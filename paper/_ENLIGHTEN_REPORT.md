# ENLighten Compressor Audit & Hardening Report

**Module:** `Codes/framework/compression_engine/enlighten_compressor.py`  
**Registry Names:** `enlighten`, `enlighten_compressor`  
**Test Suite:** `Codes/framework/tests/test_enlighten.py`  
**Date:** September 22, 2026  

---

## 1. Executive Summary

The ENLighten compressor module decomposes linear and projection weights into a low-rank SVD component combined with a 2:4 structured sparse residual:
$$W \approx U_r S_r V_r^\top + \text{Prune}_{2:4}(W - U_r S_r V_r^\top)$$

Two critical defects were identified and resolved during this audit:
1. **Missing Registry Alias:** The module registered only `@COMPRESSOR_REGISTRY.register("enlighten")`, but runner scripts and test suites often look up `"enlighten_compressor"` (such as in `verify_all_modules.py` and `run_quality_sweep.py`). This led to `KeyError` or skipped compression during automated pipeline discovery.
2. **Artificial Scale Distortion:** In `_decompose_and_prune`, the reconstructed matrix $W_{\text{recon}}$ was artificially multiplied by a Frobenius norm ratio:
   $$\text{scale} = \frac{\|W\|_F}{\|W_{\text{recon}}\|_F}$$
   $$W_{\text{new}} = W_{\text{recon}} \cdot \text{scale}$$
   This heuristic scaling amplified pruned weights and destroyed the exact zero structure of the $2:4$ sparse residual, as well as altering the optimal singular values from the truncated SVD.

---

## 2. Remediations Applied

1. **Dual Registry Alias:**
   - Added `@COMPRESSOR_REGISTRY.register("enlighten_compressor")` to both `ENLightenCompressor` and the `enlighten_factory` function.
2. **Removed Scale Distortion:**
   - Removed the artificial Frobenius scaling. The reconstructed matrix is now the exact sum $L + \text{residual}_{\text{sparse}}$, maintaining exact $2:4$ sparsity along row blocks and preserving SVD optimality.
3. **Comprehensive Verification Suite:**
   - Added `Codes/framework/tests/test_enlighten.py` covering registry lookup, block-wise 2:4 structured sparsity validation, error computation correctness, attention-only targeting, and fallback handling.

---

## 3. Test Results

The unit test suite `Codes/framework/tests/test_enlighten.py` passed all 8 tests:
1. `test_registry`: Verified both `enlighten` and `enlighten_compressor` registered in `COMPRESSOR_REGISTRY`.
2. `test_vectorized_24_pruning`: Verified that every 4-block along row columns contains at most 2 non-zero elements.
3. `test_unstructured_prune`: Verified unstructured magnitude pruning target sparsity fraction.
4. `test_enlighten_compression_forward`: Verified end-to-end model compression and forward inference pass.
5. `test_no_scale_distortion`: Verified that reconstruction error calculation matches exact Frobenius distance without artificial scaling.
6. `test_target_attention_only`: Verified attention projection isolation.
7. `test_fallback_on_unreachable_error`: Verified safe fallback to original module when error threshold is breached.
8. `test_factory_function`: Verified `enlighten_factory` invocation.

**Pass Rate:** 8/8 tests passed (100%).
