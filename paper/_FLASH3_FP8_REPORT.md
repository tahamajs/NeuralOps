# FlashAttention-3 FP8 Audit & Hardening Report

**Module:** `Codes/framework/attentions/flash_attention3_fp8.py`  
**Registry Names:** `flash3_fp8`, `flash_attention3_fp8`  
**Test Suite:** `Codes/framework/tests/test_flash3_fp8.py`  
**Date:** September 22, 2026  

---

## 1. Executive Summary

FlashAttention-3 FP8 previously had several fundamental methodology and execution flaws:
1. **Unscaled FP8 Casting:** Tensors were directly cast using `.to(torch.float8_e4m3fn)` without calculating max absolute dynamic scaling factors (`q_scale = max(|Q|) / 448.0`). This caused numerical clipping, saturation, underflow, and immediate model degradation.
2. **Missing De-scaling:** In the original FP8 attention block, output was not de-scaled by the value tensor scaling factor, distorting activation scales.
3. **Premature Hardware Assumption:** It assumed FP8 FlashAttention-3 kernels could run on any CUDA device when in fact FlashAttention-3 FP8 Tensor Core acceleration is strictly designed for Hopper (SM 9.0+) GPUs. On Ada Lovelace (SM 8.9, e.g. NVIDIA GeForce RTX 4090 / 4070) and Ampere (SM 8.0/8.6), attempting FA3 FP8 causes crashes or silent fallbacks.
4. **Missing Head Mask Support:** While SDPA handled head masking, the FP8 execution path ignored `head_mask`.
5. **Silent Missing Weight Fallback:** In `copy_weights`, missing mandatory projections (Q/K/V) only logged a debug message rather than raising an error, risking uninitialized weights in downstream benchmarks.

These issues have been resolved. The module now features:
- Per-tensor dynamic scaling to FP8 max representable magnitude ($448.0$) for Q, K, and V.
- Exact output de-scaling ($V_{\text{scale}}$).
- Hardware architecture gating (`cc[0] >= 9` check) with automated fallback to native PyTorch `F.scaled_dot_product_attention` on pre-Hopper GPUs and CPU.
- `use_fp8=False` by default to prevent unexpected hardware crashes.
- Proper `head_mask` application in all execution paths.
- Strict `RuntimeError` raising on missing mandatory weight projections.

---

## 2. Methodology & Honesty Statement

> **Important Honesty Notice:**  
> FlashAttention-3 FP8 is a **hardware execution acceleration technique**, NOT a model compression technique.  
> It does **not** compress model parameters or save persistent disk storage (unless auxiliary LoRA adapters are configured). Base weights are retained in original precision, and inputs are dynamically scaled and quantized during the attention forward pass on Hopper GPUs.

---

## 3. Hardware Support Matrix

| Hardware Architecture | Compute Capability | FlashAttention-3 FP8 Supported | Fallback Path |
| :--- | :--- | :--- | :--- |
| **NVIDIA Hopper (H100, H800, H200)** | SM 9.0 / 9.0a | Yes (Native FP8 Tensor Cores) | SDPA on error |
| **NVIDIA Blackwell (B100, B200)** | SM 10.0 | Yes (FP8 / FP4 Tensor Cores) | SDPA on error |
| **NVIDIA Ada Lovelace (RTX 4090, RTX 4070, L40)** | SM 8.9 | No (Pre-Hopper) | Automated Native SDPA |
| **NVIDIA Ampere (A100, RTX 3090, RTX 3080)** | SM 8.0 / 8.6 | No (Pre-Hopper) | Automated Native SDPA |
| **Apple Silicon / CPU** | N/A | No | Automated Native SDPA |

---

## 4. Verification & Testing

The comprehensive test suite `Codes/framework/tests/test_flash3_fp8.py` verifies all 10 critical requirements:
1. `test_default_init_and_forward`: Default initialization has `use_fp8=False` and completes forward pass cleanly.
2. `test_auto_disable_fp8_on_non_hopper`: Setting `use_fp8=True` on non-Hopper hardware or CPU automatically disables `use_fp8`.
3. `test_none_config_init`: Module correctly handles `config=None` with fallback keyword dimensions.
4. `test_distilbert_weight_copy`: Exact weight copy from DistilBERT projections (`q_lin`, `k_lin`, `v_lin`, `out_lin`).
5. `test_roberta_weight_copy_and_nested_bypass`: Exact weight copy from nested RoBERTa attention and bypasses inner `out_proj`.
6. `test_gpt2_weight_copy_and_causal`: Fused `c_attn` splitting, causal mask handling, and weight copying.
7. `test_qwen_gqa_and_rope`: Grouped Query Attention (GQA) head repeat and RoPE embeddings.
8. `test_missing_weights_raises_runtime_error`: Raises `RuntimeError` when source module is missing mandatory projections.
9. `test_head_mask_handling`: Applies head masks accurately to attention outputs.
10. `test_inject_flash3_fp8`: Verifies automated layer replacement via `inject_flash3_fp8`.

**Result:** All 10 tests passed (100% pass rate).
