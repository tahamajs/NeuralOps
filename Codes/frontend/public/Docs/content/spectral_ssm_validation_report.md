# Comprehensive Validation & Audit Report: Spectral & Recurrent SSM Attention Engines

**Domain:** AEGIS Spectral Filtering, State Space Models (SSM) & Linear Recurrences  
**Framework Location:** `Codes/framework/attentions/`  
**Date:** August 2026  

---

## 1. Executive Summary
Spectral and SSM models combine $\mathcal{O}(N)$ linear sequence scaling with frequency-domain filtering. Top performers include `aegis_v4` (**84.06% Acc**), `aegis_hybrid` (**82.34% Acc**), and `gated_linear_attn` / `decay_linear_attn` (**79.82% Acc**).

---

## 2. Empirical Benchmark Verification (SST-2)

| Model / Run Name | Accuracy (%) | F1 Score (%) | Loss | Time (s) | Recurrence / Spectral Type |
|---|---|---|---|---|---|
| `aegis_v4_sst2_s42` | 84.06% | 84.05% | 0.4713 | 976.3s | Linear / Spectral |
| `aegis_hybrid_sst2_s42` | 82.34% | 82.34% | 0.4443 | 793.6s | Linear / Spectral |
| `gated_linear_attn_sst2_s42` | 79.82% | 79.76% | 0.4950 | 19.8s | Linear / Spectral |
| `decay_linear_attn_sst2_s42` | 79.82% | 79.76% | 0.4950 | 19.5s | Linear / Spectral |
| `spectra_v2_sst2_s42` | 77.06% | 76.87% | 0.6951 | 18.9s | Linear / Spectral |
| `spectron_sst2_s42` | 76.61% | 76.38% | 0.5623 | 30.6s | Linear / Spectral |
| `inject_aegis_spectron_fp8_sst2_s42` | 70.53% | 70.52% | 0.9112 | 744.4s | Linear / Spectral |
| `soa_module_sst2_s42` | 56.88% | 48.14% | 0.6588 | 8.4s | Linear / Spectral |
| `soa_attn_sst2_s42` | 56.88% | 48.14% | 0.6588 | 9.0s | Linear / Spectral |
| `soa_fused_attn_sst2_s42` | 56.88% | 48.14% | 0.6588 | 8.4s | Linear / Spectral |
| `soa_fused_module_sst2_s42` | 56.88% | 48.14% | 0.6588 | 8.3s | Linear / Spectral |
| `mamba_inject_sst2_s42` | 50.92% | 33.74% | 0.6982 | 13.9s | Linear / Spectral |
| `mamba_svd_flash_sst2_s42` | 50.92% | 33.74% | 0.7014 | 43.9s | Linear / Spectral |
| `inject_aegis_hybrid_sst2_s42` | 50.92% | 33.74% | 0.6905 | 11.1s | Linear / Spectral |
| `aegis_spectron_fp8_sst2_s42` | 50.92% | 33.74% | 0.6937 | 100.1s | Linear / Spectral |
| `hgrn2_layer_sst2_s42` | 50.92% | 33.74% | 0.6936 | 55.4s | Linear / Spectral |
| `factorised_matrix_state_sst2_s42` | 50.92% | 33.74% | 0.6934 | 789.0s | Linear / Spectral |
| `aegis_spectron_fp8_attn_sst2_s42` | 50.92% | 33.74% | 0.6937 | 102.2s | Linear / Spectral |
| `moe_deltakv_paged_sst2_s42` | 50.92% | 33.74% | 0.6982 | 53.9s | Linear / Spectral |
| `hgrn2_gated_deltanet_sst2_s42` | 50.92% | 33.74% | 0.6929 | 133.9s | Linear / Spectral |
| `gated_delta_net_sst2_s42` | 50.92% | 33.74% | 0.6950 | 63.0s | Linear / Spectral |
| `aegis_v3_sst2_s42` | 50.92% | 33.74% | 0.6945 | 10.2s | Linear / Spectral |
| `kda_sst2_s42` | 50.92% | 33.74% | 0.6966 | 380.7s | Linear / Spectral |
| `hla_sst2_s42` | 50.92% | 33.74% | 0.7025 | 30.3s | Linear / Spectral |
| `spectral_mamba_sst2_s42` | 50.92% | 33.74% | 0.6972 | 84.7s | Linear / Spectral |
| `griffin_ssd_sst2_s42` | 50.92% | 33.74% | 0.6988 | 48.8s | Linear / Spectral |
| `gated_deltanet_block_sst2_s42` | 50.92% | 33.74% | 0.7084 | 29.2s | Linear / Spectral |
| `aegis_hybrid_attn_sst2_s42` | 50.92% | 33.74% | 0.7027 | 8.7s | Linear / Spectral |
| `gated_delta_net_inject_sst2_s42` | 50.92% | 33.74% | 0.7084 | 33.2s | Linear / Spectral |
| `simple_ssm_sst2_s42` | 50.92% | 33.74% | 0.7026 | 51.9s | Linear / Spectral |
| `aegis_v6_sst2_s42` | 49.08% | 32.92% | 0.6958 | 1099.3s | Linear / Spectral |
| `aegis_v2_sst2_s42` | 49.08% | 32.92% | nan | 713.3s | Linear / Spectral |
| `aegis_v5_sst2_s42` | 48.05% | 36.72% | nan | 733.7s | Linear / Spectral |

---

## 3. Analysis: Why Unidirectional SSMs Stalled at 50.92% & The Bidirectional Fix
In bidirectional classification tasks (e.g. SST-2 sentiment analysis), tokens depend on both left and right context.
- **Root Cause of 50.92% Baseline:** Pure causal recurrence $h_t = A h_{t-1} + B x_t$ prevents the `[CLS]` token (index 0) or early tokens from observing future sentimental markers.
- **The Architectural Fix:**
  1. **Bidirectional Chunk Scanning:** Implement dual forward-backward recurrence passes:
     $$h_t^{\text{fwd}} = A_{\text{fwd}} h_{t-1} + B_{\text{fwd}} x_t, \quad h_t^{\text{bwd}} = A_{\text{bwd}} h_{t+1} + B_{\text{bwd}} x_t$$
     $$y_t = W_{\text{out}} [h_t^{\text{fwd}} ; h_t^{\text{bwd}}]$$
  2. **Gating Bias Warmup:** Initialize AEGIS gating biases to `gate_bias = +2.0` so the network begins training on the full attention path and progressively blends spectral filters.

---

## 4. Key Recommendations
1. **Learnable Log-Decay Parameterization:** Bound recurrence decay factors $\gamma_t = \sigma(\lambda_t) \in (0, 1)$ with learnable log-decay initialization to avoid gradient explosion.
2. **Spectral DCT Normalization:** Apply orthonormal Discrete Cosine Transform (DCT-II) filtering with high-frequency attenuation to capture long-range semantic coherence.
