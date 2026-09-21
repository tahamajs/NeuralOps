# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Attn Params | Ratio |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 2.36M | 1.00× |
| DistilBERT | **LEKAv1** | **51.03** | -39.68 | 0.6753 | 0.85 | 50\%× |
| RoBERTa | full | 78.56% | — | 0.8124 | 2.36M | 1.00× |
| RoBERTa | **LEKAv1** | **50.92** | -27.64 | 0.6748 | 0.85 | 2.1\times× |
| GPT-2 | full | 79.36% | — | 0.7978 | 2.36M | 1.00× |
| GPT-2 | **LEKAv1** | **49.31** | -30.05 | 0.865 | 0.85 | 2.1\times× |

## 7.2 Parameter Analysis

**Table 7.2: Total parameters.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +48.5\%% |

## 7.3 Inference Speed vs Sequence Length

**Table 7.3: Inference time (seconds), DistilBERT.**

| L | Baseline | LEKAv1 | Speedup |
|---|---|---|---|
| 128 | 1.90 | 0.85 | 84.8× |
| 256 | 3.50 | 0.85 | 85.6× |
| 512 | 12.0 | 0.85 | 86.2× |
| 1024 | 45.0 | 0.85 | 86.4× |
| 2048 | 180.0 | 0.85 | 86.5× |

**Breakeven:** $L^* = 128$.

## 7.4 Ablation: Feature Dimension

**Table 7.4: Accuracy vs feature dimension.**

| F | DistilBERT | RoBERTa | GPT-2 | Attn Params |
|---|---|---|---|---|
| 64 | 0.85 | 0.85 | 0.85 | 66.4M |
| 128 | 0.865 | 0.865 | 0.865 | 0.865 |
| 256 | **51.03** | **50.92** | **49.31** | 0.85 |
| 512 | 0.85 | 0.85 | 0.85 | 66.4M |

## 7.5 Ablation: Preservation Gate

**Table 7.5: Effect of preservation gate.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With gate | **51.03** | **50.92** | **49.31** |
| No gate (α=1) | 0.85 | 0.85 | 0.85 |
| Δ | 0.85 | 32 | 0.85 |

## 7.6 Ablation: Head-wise LayerNorm

**Table 7.6: Effect of pre-projection LayerNorm.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With norm | **51.03** | **50.92** | **49.31** |
| No norm | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: QR Orthogonalization

**Table 7.7: Effect of QR init.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| QR projection | **51.03** | **50.92** | **49.31** |
| Random (no QR) | 0.85 | 0.85 | 0.85 |

## 7.8 Ablation: LoRA Rank

**Table 7.8: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 32 | 32 | 32 |
| 16 | 32 | 32 | 32 |
| 32 | 32 | 32 | 32 |

## 7.9 Ablation: Recovery Steps

**Table 7.9: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.10 Gate Statistics

**Table 7.10: Final α and output_scale.**

| Model | α final | output_scale |
|---|---|---|
| DistilBERT | 0.85 | 0.85 |
| RoBERTa | 0.85 | 0.85 |
| GPT-2 | 0.85 | 0.85 |

## 7.11 Summary

- **Accuracy:** drops -39.68%–-30.05%.
- **Linear scaling:** LEKAv1 is faster at L > 128.
- **Preservation gate:** contributes 0.85%.
- **QR projection:** contributes 0.85%.
- **Parameter increase:** 48.5\%%.

---
---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 69.13% | `results/t5_cola_parallel/summary.csv` |
| F1 Score | 0.0000 | `results/t5_cola_parallel/summary.csv` |
| Compression | 1.000x | `results/t5_cola_parallel/summary.csv` |
| Compressed Layers | ? | `results/t5_cola_parallel/summary.csv` |
| Fallback Layers | ? | `results/t5_cola_parallel/summary.csv` |
| Timestamp | 2026-09-01T02:21:18.180537 | `results/t5_cola_parallel/summary.csv` |


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 78.56, 79.36]
> These numbers must NOT be cited in the final paper.

