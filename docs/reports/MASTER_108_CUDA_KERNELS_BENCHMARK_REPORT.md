# 🔬 NEURALOPS v1.3.7 — RIGOROUS FORENSIC BENCHMARK AUDIT & CLASSIFICATION REPORT

**Evaluation Target**: NVIDIA GeForce RTX 4070 Laptop GPU (Ada Lovelace, SM 8.9, 8GB VRAM)  
**Execution Environment**: Python 3.10 / 3.14 + PyTorch 2.x CUDA 12.8  
**Audit Scope**: All Native CUDA C++ Modules in `neuralops/core/cuda_kernels/`  
**Classification Protocol**: Strict Multi-Metric Forensic Audit (No Silent Pass, No Fallback Masking)

---

## 📊 EXECUTIVE SUMMARY & FORENSIC METRICS

An exhaustive forensic audit of the raw benchmark telemetry across all 108 native CUDA modules reveals that the previous claim of 100% "PASS" was based on a naive "ran without process crash" criterion. When evaluated against rigorous systems engineering criteria:
- **Baseline Validity**: $T_{\text{ref}} \ge 0.01\text{ ms}$
- **True Hardware Speedup**: $S = T_{\text{ref}} / T_{\text{cuda}} \ge 1.0\times$
- **Numerical Parity (FP16)**: $\Delta_{\max} \le 0.01$
- **Production Latency Envelope**: $T_{\text{cuda}} \le 20.0\text{ ms}$
- **Finite Float Invariant**: $\Delta_{\max} \ne \text{NaN}$

Only **16 of 108 modules (14.8%)** genuinely pass all criteria.

| Metric / Category | Raw Execution Count | Genuine Correct Status | Forensic Diagnosis & Mandatory Remediation Action |
|:---|:---:|:---:|:---|
| **Total Evaluated CUDA Modules** | 108 / 108 | 108 / 108 Compiled | Direct C++ extension compilation verified via JIT |
| **🟢 Genuine PASS Kernels** | 16 | **16 (14.8%)** | $T_{\text{ref}} > 0.01\text{ms}$, $S \ge 1.0\times$, $\Delta_{\max} < 0.01$, $T_{\text{cuda}} < 20\text{ms}$ |
| **🟡 Borderline Passing Kernels** | 4 | **4 (3.7%)** | $S > 1.0\times$, $0.01 < \Delta_{\max} \le 0.065$ (FP32 Accumulator tightening needed) |
| **🔴 Catastrophic NaN Kernels** | 5 | **0 (0.0%)** | Produced NaN output during evaluation (**DELETE_IMMEDIATELY**) |
| **🔴 Slower Than PyTorch (< 1.0x)** | 34 | **0 (0.0%)** | 14 critical (<0.2x, up to 100x slower) + 20 moderate (<1.0x) |
| **🔴 Numerically Broken (`diff > 0.01`)** | 18 | **0 (0.0%)** | Numerical divergence vs PyTorch reference (**DEBUG_4_HOURS OR DELETE**) |
| **🔴 Missing PyTorch Baseline (`Ref=0`)** | 33 | **0 (0.0%)** | Unverifiable relative speedup (**ADD_PYTORCH_REF**) |
| **🔴 Impossible Speedups (> 100x)** | 17 | **0 (0.0%)** | Unvectorized naive Python reference loop (**FIX_PYTORCH_BASELINE**) |
| **🔴 Absolute Latency Outliers (> 20ms)** | 8 | **0 (0.0%)** | Unacceptable inference latency (**REDESIGN_OR_DELETE**) |

---

## 📋 FIVE-PHASE REMEDIATION ARCHITECTURE

1. **Phase 1 (Harness Lockdown)**: Implemented strict `classify_status()` logic directly into `scripts/benchmarks/benchmark_all_108_cuda_isolated.py` with the `--classify-strict` CLI flag, preventing any unearned "PASS" statuses from being written to telemetry files.
2. **Phase 2 (Strict Multi-Metric Benchmark Sweep)**: Executed subprocess-isolated sweeps on the remote RTX 4070 GPU, capturing unmasked latency, reference timing, speedup, and numerical error.
3. **Phase 3 (Preserve & Protect Genuine Kernels)**: Locked down regression parity testing for the 16 genuine passing modules.
4. **Phase 4 (Targeted Remediation & Deprecation Triage)**: Classified the 5 NaN, 18 numerical mismatch, and 34 slower-than-PyTorch kernels into strict action buckets (Immediate Deletion, 4-Hour Debugging Box, or Vectorized Baseline Rewriting).
5. **Phase 5 (Reporting & Audit Persistence)**: Documented all 7 forensic tables in the repository.

---

### Table 1: The 16 Genuine PASS Kernels

| # | Module | CUDA Latency (ms) | Reference Latency (ms) | Speedup vs Ref | Max Absolute Error | Recommendation |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `block_diag_gemm` | 0.0185 | 0.0236 | 1.27x | 0.0078 | **KEEP** |
| 2 | `entropy_pruning` | 0.0084 | 0.0311 | 3.71x | 0.0000 | **KEEP** |
| 3 | `fast_gelu` | 0.0082 | 0.0282 | 3.43x | 0.0020 | **KEEP** |
| 4 | `flash_gqa` | 0.4708 | 1.2731 | 2.70x | 0.0002 | **KEEP** |
| 5 | `fused_kd` | 0.0370 | 0.0389 | 1.05x | 0.0000 | **KEEP** |
| 6 | `fused_rope` | 0.0220 | 0.0395 | 1.80x | 0.0078 | **KEEP** |
| 7 | `jamba` | 0.0064 | 0.0179 | 2.81x | 0.0020 | **KEEP** |
| 8 | `layernorm` | 0.0093 | 0.0558 | 6.02x | 0.0020 | **KEEP** |
| 9 | `paged_kv_cache` | 0.0053 | 0.0707 | 13.33x | 0.0000 | **KEEP** |
| 10 | `paged_kv_evict` | 0.0070 | 0.0098 | 1.41x | 0.0000 | **KEEP** |
| 11 | `recurrent_gla_step` | 0.0165 | 0.0629 | 3.81x | 0.0020 | **KEEP** |
| 12 | `rmsnorm` | 0.0092 | 0.0377 | 4.12x | 0.0002 | **KEEP** |
| 13 | `sliding_window` | 0.6016 | 1.8778 | 3.12x | 0.0005 | **KEEP** |
| 14 | `spectraformer_attention` | 0.0068 | 0.0105 | 1.54x | 0.0078 | **KEEP** |
| 15 | `swiglu` | 0.0081 | 0.0102 | 1.26x | 0.0039 | **KEEP** |
| 16 | `ternary_gemm` | 0.0142 | 0.1750 | 12.31x | 0.0039 | **KEEP** |

---

### Table 1B: Borderline Passing Kernels (0.01 < max_diff <= 0.065, speedup > 1.0x)

| # | Module | CUDA Latency (ms) | Reference Latency (ms) | Speedup vs Ref | Max Absolute Error | Recommendation |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `kronecker_gemm` | 0.0109 | 0.0273 | 2.50x | 0.0625 | **KEEP_BORDERLINE (TIGHTEN_NUMERICS)** |
| 2 | `recurrent_hgrn2_step` | 0.0107 | 0.0177 | 1.66x | 0.0156 | **KEEP_BORDERLINE (TIGHTEN_NUMERICS)** |
| 3 | `soa_attn` | 0.0110 | 0.0349 | 3.17x | 0.0625 | **KEEP_BORDERLINE (TIGHTEN_NUMERICS)** |
| 4 | `state_update` | 0.0185 | 0.0898 | 4.86x | 0.0312 | **KEEP_BORDERLINE (TIGHTEN_NUMERICS)** |

---

### Table 2: The 34 Slower-Than-PyTorch Kernels (Speedup < 1.0x)

| # | Module | CUDA Latency (ms) | Reference Latency (ms) | Speedup vs Ref | Max Absolute Error | Recommendation |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `fused_ce` | 5.8149 | 0.0569 | 0.01x | 0.0000 | **DELETE** |
| 2 | `varlen_attention` | 24.4750 | 0.3285 | 0.01x | 0.0002 | **DELETE** |
| 3 | `logcumsumexp` | 0.8199 | 0.0156 | 0.02x | 0.0000 | **DELETE** |
| 4 | `split_head_norm` | 1.6739 | 0.0283 | 0.02x | 0.0000 | **DELETE** |
| 5 | `rotational_svd` | 1.1070 | 0.0231 | 0.02x | 2.0000 | **DELETE** |
| 6 | `jamba_info` | 0.3113 | 0.0114 | 0.04x | 0.0078 | **DELETE** |
| 7 | `nystrom_attention` | 0.3152 | 0.0152 | 0.05x | 0.1250 | **DELETE** |
| 8 | `rpca_sparse_svd` | 0.2994 | 0.0157 | 0.05x | 0.1250 | **DELETE** |
| 9 | `svd_prefix_scan` | 49.5671 | 5.4709 | 0.11x | 0.0000 | **DELETE** |
| 10 | `flash_decoding` | 4.2425 | 0.6210 | 0.15x | 0.0017 | **DELETE** |
| 11 | `gla_scan` | 144.5527 | 22.6602 | 0.16x | 0.0312 | **DELETE** |
| 12 | `cosformer` | 8.3855 | 1.4652 | 0.17x | 0.0001 | **DELETE** |
| 13 | `routing_attention` | 3.3865 | 0.5645 | 0.17x | 0.0007 | **DELETE** |
| 14 | `sparse_attention` | 3.3353 | 0.5702 | 0.17x | 0.0010 | **DELETE** |
| 15 | `act_scale_gemm` | 0.0335 | 0.0295 | 0.88x | 0.0625 | **PROFILE_AND_FIX** |
| 16 | `alibi_attention` | 4.8804 | 1.3313 | 0.27x | 0.0010 | **PROFILE_AND_FIX** |
| 17 | `bigbird` | 2.0251 | 0.5115 | 0.25x | 0.0020 | **PROFILE_AND_FIX** |
| 18 | `bitonic_topk` | 0.0268 | 0.0175 | 0.65x | 778.0000 | **PROFILE_AND_FIX** |
| 19 | `channel_shuffle` | 0.0095 | 0.0094 | 0.98x | 0.0000 | **PROFILE_AND_FIX** |
| 20 | `cross_attention` | 4.2521 | 0.9820 | 0.23x | 0.0002 | **PROFILE_AND_FIX** |
| 21 | `fp4_gemm` | 0.1218 | 0.0817 | 0.67x | 0.2500 | **PROFILE_AND_FIX** |
| 22 | `galerkin_attention` | 0.0476 | 0.0274 | 0.58x | 0.0020 | **PROFILE_AND_FIX** |
| 23 | `hydra_attention` | 0.0478 | 0.0259 | 0.54x | 0.0005 | **PROFILE_AND_FIX** |
| 24 | `hymba_transx` | 0.0337 | 0.0152 | 0.45x | 0.0020 | **PROFILE_AND_FIX** |
| 25 | `local_attention` | 0.7634 | 0.4695 | 0.61x | 0.0015 | **PROFILE_AND_FIX** |
| 26 | `longformer` | 1.0369 | 0.4868 | 0.47x | 0.0015 | **PROFILE_AND_FIX** |
| 27 | `reformer` | 0.7538 | 0.5373 | 0.71x | 0.0039 | **PROFILE_AND_FIX** |
| 28 | `samba_moe` | 0.0225 | 0.0140 | 0.62x | 19.3125 | **PROFILE_AND_FIX** |
| 29 | `softplus` | 0.0080 | 0.0050 | 0.62x | 0.0000 | **PROFILE_AND_FIX** |
| 30 | `spectral_mamba` | 0.0358 | 0.0183 | 0.51x | 0.0000 | **PROFILE_AND_FIX** |
| 31 | `spectral_shrinkage` | 0.2636 | 0.2068 | 0.78x | 0.0000 | **PROFILE_AND_FIX** |
| 32 | `speculative_verify` | 0.0335 | 0.0269 | 0.80x | 0.0000 | **PROFILE_AND_FIX** |
| 33 | `tpa_attention` | 0.0252 | 0.0089 | 0.35x | 0.0312 | **PROFILE_AND_FIX** |
| 34 | `upi_stuffed_mamba` | 0.0268 | 0.0140 | 0.52x | 0.0020 | **PROFILE_AND_FIX** |

---

### Table 3: The 5 Catastrophic NaN Kernels

| # | Module | CUDA Latency (ms) | Reference Latency (ms) | Speedup vs Ref | Max Absolute Error | Recommendation |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `flash_decode_stage2` | 0.0061 | 0.0467 | 7.67x | NaN | **DELETE_IMMEDIATELY** |
| 2 | `linear_attention` | 3.1336 | 887.2009 | 283.12x | NaN | **DELETE_IMMEDIATELY** |
| 3 | `rfa_attention` | 0.0322 | 15.4039 | 477.73x | NaN | **DELETE_IMMEDIATELY** |
| 4 | `selective_scan` | 0.3558 | 27.6389 | 77.69x | NaN | **DELETE_IMMEDIATELY** |
| 5 | `selective_ssm` | 0.0368 | 18.9747 | 516.20x | NaN | **DELETE_IMMEDIATELY** |

---

### Table 4: The 18 Numerically Broken Kernels (max_diff > 0.01)

| # | Module | CUDA Latency (ms) | Reference Latency (ms) | Speedup vs Ref | Max Absolute Error | Recommendation |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `bitonic_topk` | 0.0268 | 0.0175 | 0.65x | 778.0000 | **DELETE** |
| 2 | `simple_ssm` | 0.0295 | 8.5663 | 290.44x | 493.5000 | **DELETE** |
| 3 | `lut_gemm` | 0.0240 | 0.1007 | 4.20x | 53.0234 | **DELETE** |
| 4 | `samba_moe` | 0.0225 | 0.0140 | 0.62x | 19.3125 | **DELETE** |
| 5 | `nasc_attention` | 0.0329 | 11.8186 | 358.85x | 2.5000 | **DELETE** |
| 6 | `rotational_svd` | 1.1070 | 0.0231 | 0.02x | 2.0000 | **DELETE** |
| 7 | `vspec` | 0.0360 | 15.6062 | 434.09x | 1.2031 | **DELETE** |
| 8 | `linear_rnn_attention` | 0.0315 | 9.1643 | 290.89x | 1.0000 | **DEBUG_4_HOURS** |
| 9 | `samba_m3` | 0.0105 | 0.0239 | 2.27x | 0.7322 | **DEBUG_4_HOURS** |
| 10 | `goldfinch_ahn` | 0.0328 | 15.8995 | 484.99x | 0.5000 | **DEBUG_4_HOURS** |
| 11 | `orion` | 0.0337 | 9.5252 | 282.77x | 0.3438 | **DEBUG_4_HOURS** |
| 12 | `decay_linear_attention` | 0.0317 | 11.6290 | 366.66x | 0.3125 | **DEBUG_4_HOURS** |
| 13 | `nexus` | 0.0340 | 9.5726 | 281.27x | 0.2812 | **DEBUG_4_HOURS** |
| 14 | `fp4_gemm` | 0.1218 | 0.0817 | 0.67x | 0.2500 | **DEBUG_4_HOURS** |
| 15 | `hgrn2_scan` | 0.0849 | 22.8187 | 268.62x | 0.2500 | **DEBUG_4_HOURS** |
| 16 | `rwkv6` | 0.0390 | 15.7578 | 403.73x | 0.1250 | **DEBUG_4_HOURS** |
| 17 | `nystrom_attention` | 0.3152 | 0.0152 | 0.05x | 0.1250 | **DEBUG_4_HOURS** |
| 18 | `rpca_sparse_svd` | 0.2994 | 0.0157 | 0.05x | 0.1250 | **DEBUG_4_HOURS** |

---

### Table 5: The 33 No-Baseline Kernels (Ref = 0.0 ms)

| # | Module | CUDA Latency (ms) | Reference Latency (ms) | Speedup vs Ref | Max Absolute Error | Recommendation |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `adaptive_span` | 2.4958 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 2 | `adasplash_attention` | 326.7335 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 3 | `aegis_gate_norm` | 5.9491 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 4 | `causal_conv1d` | 0.0068 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 5 | `cluster_attention` | 0.2028 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 6 | `differential_attention` | 2.7716 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 7 | `dynamic_conv1d` | 24.2119 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 8 | `entmax_bisect` | 2.7052 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 9 | `entropy_gate` | 0.2119 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 10 | `favor_plus` | 0.0102 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 11 | `fms_scan` | 142.3113 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 12 | `fused_dropout_add` | 0.0083 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 13 | `fused_norm_linear` | 0.1536 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 14 | `fwht_transform` | 0.0077 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 15 | `gdn_delta_rule` | 1.8696 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 16 | `gdn_scan` | 4.7002 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 17 | `gla_unnormalized_scan` | 0.0333 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 18 | `hadamard_gemm` | 6.1501 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 19 | `hstu_attention` | 0.0321 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 20 | `linformer_attention` | 59.3174 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 21 | `low_rank_svd` | 0.0205 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 22 | `mamba_block` | 0.1596 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 23 | `mamba_svd_flash` | 0.8673 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 24 | `moe_routing` | 0.0268 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 25 | `multi_scale_retention` | 146.5290 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 26 | `paged_attention` | 0.0394 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 27 | `paged_recurrent_step` | 30.6269 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 28 | `phantom` | 0.0068 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 29 | `quantized_gemm` | 0.0970 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 30 | `rms_variance_match` | 5.3562 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 31 | `spectra_v2` | 0.0064 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 32 | `tt_contraction` | 0.0074 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |
| 33 | `unified_rope_offset` | 6.0858 | N/A | 1.00x | N/A | **ADD_PYTORCH_REF** |

---

### Table 6: The 17 Impossible Speedup Kernels (Speedup > 100x)

| # | Module | CUDA Latency (ms) | Reference Latency (ms) | Speedup vs Ref | Max Absolute Error | Recommendation |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `selective_ssm` | 0.0368 | 18.9747 | 516.20x | NaN | **FIX_PYTORCH_BASELINE** |
| 2 | `goldfinch_ahn` | 0.0328 | 15.8995 | 484.99x | 0.5000 | **FIX_PYTORCH_BASELINE** |
| 3 | `rfa_attention` | 0.0322 | 15.4039 | 477.73x | NaN | **FIX_PYTORCH_BASELINE** |
| 4 | `fast_attention` | 0.0329 | 15.4691 | 470.44x | 0.0078 | **FIX_PYTORCH_BASELINE** |
| 5 | `vspec` | 0.0360 | 15.6062 | 434.09x | 1.2031 | **FIX_PYTORCH_BASELINE** |
| 6 | `rwkv6` | 0.0390 | 15.7578 | 403.73x | 0.1250 | **FIX_PYTORCH_BASELINE** |
| 7 | `griffin_ssd` | 0.0326 | 12.9670 | 397.95x | 0.0029 | **FIX_PYTORCH_BASELINE** |
| 8 | `decay_linear_attention` | 0.0317 | 11.6290 | 366.66x | 0.3125 | **FIX_PYTORCH_BASELINE** |
| 9 | `nasc_attention` | 0.0329 | 11.8186 | 358.85x | 2.5000 | **FIX_PYTORCH_BASELINE** |
| 10 | `hgrn2_spatial_scan` | 0.0330 | 10.7125 | 324.58x | 0.0039 | **FIX_PYTORCH_BASELINE** |
| 11 | `linear_rnn_attention` | 0.0315 | 9.1643 | 290.89x | 1.0000 | **FIX_PYTORCH_BASELINE** |
| 12 | `simple_ssm` | 0.0295 | 8.5663 | 290.44x | 493.5000 | **FIX_PYTORCH_BASELINE** |
| 13 | `linear_attention` | 3.1336 | 887.2009 | 283.12x | NaN | **FIX_PYTORCH_BASELINE** |
| 14 | `orion` | 0.0337 | 9.5252 | 282.77x | 0.3438 | **FIX_PYTORCH_BASELINE** |
| 15 | `nexus` | 0.0340 | 9.5726 | 281.27x | 0.2812 | **FIX_PYTORCH_BASELINE** |
| 16 | `hgrn2_scan` | 0.0849 | 22.8187 | 268.62x | 0.2500 | **FIX_PYTORCH_BASELINE** |
| 17 | `selective_scan` | 0.3558 | 27.6389 | 77.69x | NaN | **FIX_PYTORCH_BASELINE** |

---

### Table 7: The 8 Latency Outliers (> 20.0 ms)

| # | Module | CUDA Latency (ms) | Reference Latency (ms) | Speedup vs Ref | Max Absolute Error | Recommendation |
|:-:|:---|:---:|:---:|:---:|:---:|:---:|
| 1 | `adasplash_attention` | 326.7335 | N/A | 1.00x | N/A | **REDESIGN_OR_DELETE** |
| 2 | `multi_scale_retention` | 146.5290 | N/A | 1.00x | N/A | **REDESIGN_OR_DELETE** |
| 3 | `gla_scan` | 144.5527 | 22.6602 | 0.16x | 0.0312 | **REDESIGN_OR_DELETE** |
| 4 | `fms_scan` | 142.3113 | N/A | 1.00x | N/A | **REDESIGN_OR_DELETE** |
| 5 | `svd_prefix_scan` | 49.5671 | 5.4709 | 0.11x | 0.0000 | **REDESIGN_OR_DELETE** |
| 6 | `paged_recurrent_step` | 30.6269 | N/A | 1.00x | N/A | **REDESIGN_OR_DELETE** |
| 7 | `varlen_attention` | 24.4750 | 0.3285 | 0.01x | 0.0002 | **REDESIGN_OR_DELETE** |
| 8 | `dynamic_conv1d` | 24.2119 | N/A | 1.00x | N/A | **REDESIGN_OR_DELETE** |
