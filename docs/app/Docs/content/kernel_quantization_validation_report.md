# Comprehensive Validation & Audit Report: Kernel, Sparse & Quantization Engines

**Domain:** Kernel Random Features, Sparse Landmark Attention & Quantization / KV Cache Compression  
**Framework Location:** `Codes/framework/attentions/` and `Codes/framework/compression_engine/`  
**Date:** August 2026  

---

## 1. Executive Summary
Kernel and Sparse methods approximate the full attention softmax matrix $\operatorname{Softmax}(Q K^T / \sqrt{d})$ with sub-quadratic complexity. Top performers include `span_expanded` (**81.31% Acc**), `adaptive_span` (**80.39% Acc in only 5.3s**), `xattention` (**79.24% Acc**), and `salt_attention` (**77.75% Acc**).

---

## 2. Empirical Benchmark Verification (SST-2)

| Model / Run Name | Accuracy (%) | F1 Score (%) | Loss | Time (s) | Compression / Approximation Method |
|---|---|---|---|---|---|
| `span_expanded_sst2_s42` | 81.31% | 81.31% | 0.4050 | 95.6s | Kernel / Sparse / Quant |
| `adaptive_span_sst2_s42` | 80.39% | 80.37% | 0.5177 | 5.3s | Kernel / Sparse / Quant |
| `xattention_sst2_s42` | 79.24% | 78.91% | 0.5323 | 19.0s | Kernel / Sparse / Quant |
| `salt_attention_sst2_s42` | 77.75% | 77.62% | 0.5879 | 26.4s | Kernel / Sparse / Quant |
| `lsh_sst2_s42` | 76.49% | 76.48% | 0.5405 | 25163.9s | Kernel / Sparse / Quant |
| `nsa_sst2_s42` | 74.66% | 74.42% | 0.5287 | 24.7s | Kernel / Sparse / Quant |
| `routing_sst2_s42` | 72.13% | 72.06% | 0.5916 | 26.1s | Kernel / Sparse / Quant |
| `flash3_fp8_sst2_s42` | 69.27% | 68.45% | 0.6797 | 16.1s | Kernel / Sparse / Quant |
| `cluster_sst2_s42` | 61.01% | 60.88% | 0.6641 | 8432.3s | Kernel / Sparse / Quant |
| `galerkin_sst2_s42` | 51.03% | 34.41% | 0.7057 | 4.7s | Kernel / Sparse / Quant |
| `mamba_svd_flash_sst2_s42` | 50.92% | 33.74% | 0.7014 | 43.9s | Kernel / Sparse / Quant |
| `hamming_sst2_s42` | 49.08% | 32.92% | nan | 11.9s | Kernel / Sparse / Quant |
| `adasplash_sst2_s42` | 49.08% | 32.92% | 0.6947 | 1700.1s | Kernel / Sparse / Quant |
| `mixattention_moska_sst2_s42` | 49.08% | 32.92% | nan | 11.2s | Kernel / Sparse / Quant |

---

## 3. Mathematical Foundations: Random Feature Maps & Quantization STE
### FAVOR+ Positive Random Features:
$$\operatorname{Attn}(Q, K, V) = D^{-1} (\phi(Q) (\phi(K)^\top V))$$
where $\phi(x) = \frac{1}{\sqrt{M}} \exp\left(-\frac{\|x\|^2}{2}\right) [\exp(w_1^\top x), \dots, \exp(w_M^\top x)]^\top$.

### Straight-Through Estimator (STE) for INT4 / INT8 Quantization:
$$W_q = s \cdot \operatorname{clamp}\left(\left\lfloor \frac{W}{s} \right\rceil, -2^{b-1}, 2^{b-1}-1\right)$$
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial W_q} \cdot \mathbb{I}\left(-2^{b-1} \le \frac{W}{s} \le 2^{b-1}-1\right)$$

---

## 4. Key Recommendations
1. **Orthogonal Random Features (ORF):** Use Gram-Schmidt orthogonalized Gaussian matrices $W_{\text{rff}}$ to reduce Monte-Carlo kernel variance by up to $4\times$.
2. **Sink-Aware Dynamic KV Eviction:** In ZeroKV-Neo and CacheNotes, pin the first 4 attention sinks and last 32 sliding window tokens while dynamically evicting non-heavy hitters to maintain constant $\mathcal{O}(1)$ memory.
