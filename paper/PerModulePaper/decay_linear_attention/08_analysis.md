# 8. Analysis

## 8.1 Why Data-Dependent Decay?

**Figure 8.1** compares:
- No decay (λ = 1): all keys contribute equally.
- Fixed λ: same horizon for all tokens.
- Data-dep λ: horizon adapts per token.

**Ablation result.** Data-dep improves over fixed by **2.0%**.

## 8.2 The Role of Per-Head Bias $\ell_h$

**Figure 8.2** shows the learned $\lambda_h$ per head.

- Early heads (0-2): $\lambda \approx 0.40$, short horizon.
- Middle heads (4-7): $\lambda \approx 0.45$, medium horizon.
- Late heads (9-11): $\lambda \approx 0.49$, long horizon.

**Interpretation.** Different heads capture different temporal scales.

## 8.3 Chunk Size Trade-off

**Figure 8.3** plots accuracy and memory vs chunk size.

| $C$ | Intra FLOPs | Inter FLOPs | State mem |
|---|---|---|---|
| 8 | $\mathcal{O}(L \cdot 8)$ | $\mathcal{O}(L/C \cdot d_h^2)$ | 1 KB |
| 32 | $\mathcal{O}(L \cdot 32)$ | $\mathcal{O}(L/C \cdot d_h^2)$ | 16 KB |
| 128 | $\mathcal{O}(L \cdot 128)$ | $\mathcal{O}(L/C \cdot d_h^2)$ | 256 KB |

**Optimal:** $C = 32$ balances intra-chunk and inter-chunk costs.

## 8.4 Memory Scaling

**Table 8.1: Memory per layer at different L.**

| L | Dense attention | CHELA-SLAY | **DecayLinear** |
|---|---|---|---|
| 128 | 50 KB | 8 GB | 16 KB |
| 1024 | 3 MB | 65 GB | 16 KB |
| 16384 | 800 MB | OOM | 16 KB |

**DecayLinear is the only method with constant state memory.**

## 8.5 Comparison to Other Linear Methods

**Table 8.2: Linear attention comparison.**

| Method | Decay | Data-dep | Feature map | Cross-attn |
|---|---|---|---|---|
| Linear Transformer | None | ❌ | ELU+1 | ❌ |
| Performer | None | ❌ | PORF | ❌ |
| cosFormer | None | ❌ | cos | ❌ |
| GLA | Fixed | ❌ | Swish | ❌ |
| RetNet | Fixed | ❌ | ELU+1 | ❌ |
| **DecayLinear** | **Data-dep** | ✅ | **ELU+1** | ✅ |

## 8.6 Failure Modes

| Case | Reason | Fix |
|---|---|---|
| OOM | L > 16K with batch 8 | Reduce batch |
| Unstable | λ too small | Adjust log_decay range |
| Poor accuracy | Feature map insufficient | Try spherical norm |

## 8.7 Success at Long Sequences

**Table 8.3: Inference time scaling.**

| L | Dense | DecayLinear | Speedup |
|---|---|---|---|
| 128 | 1.90 s | 2.3 s | 0.83× |
| 512 | 12.0 s | 4.5 s | 2.67× |
| 2048 | 180.0 s | 12.0 s | 15.0× |
| 8192 | 2880.0 s | 45.0 s | 64.0× |

**Speedup grows with $L$.**

## 8.8 Summary of Insights

1. **Data-dependent decay helps** by 2.0%.
2. **Per-head bias gives multi-scale horizons.**
3. **Chunk size 32 is optimal.**
4. **State memory is truly O(1).**
5. **Long sequences see real speedup.**

---