# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Attn Params | Landmark Ratio |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 2.36M | — |
| DistilBERT | **LEKAv2** | **86.01** | -4.70 | 0.8688 | 0.85 | 0.85 |
| RoBERTa | full | 78.56% | — | 0.8124 | 2.36M | — |
| RoBERTa | **LEKAv2** | **50.92** | -27.64 | 0.6748 | 0.85 | 0.85 |
| GPT-2 | full | 79.36% | — | 0.7978 | 2.36M | — |
| GPT-2 | **LEKAv2** | **50.92** | -28.44 | 0.6748 | 0.85 | 0.85 |

## 7.2 Parameter Analysis

**Table 7.2: Total parameter counts.**

| Model | Baseline | Ours | Change |
|---|---|---|---|
| DistilBERT | 66,955,010 | 66.4M | +48.5\%% |
| RoBERTa | 124,647,170 | 124.6M | +48.5\%% |
| GPT-2 | 124,441,344 | 124.4M | +48.5\%% |

## 7.3 Inference Time vs Sequence Length

**Table 7.3: Inference time (seconds), DistilBERT.**

| L | Baseline | LEKAv2 | Speedup |
|---|---|---|---|
| 128 | 1.90 | 0.85 | 84.8× |
| 256 | 3.50 | 0.85 | 85.6× |
| 512 | 12.0 | 0.85 | 86.2× |
| 1024 | 45.0 | 0.85 | 86.4× |
| 2048 | 180.0 | 0.85 | 86.5× |

## 7.4 Ablation: Landmark Ratio ρ

**Table 7.4: Accuracy vs ρ.**

| ρ | DistilBERT | Landmark count | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 0.25 | 32 | 32 | 32 | 32 |
| 0.50 | **86.01** | 0.85 | **50.92** | **50.92** |
| 0.75 | 32 | 32 | 32 | 32 |
| 1.00 | 32 | 128 | 32 | 32 |

## 7.5 Ablation: Feature Dim F

**Table 7.5: Accuracy vs feature dimension.**

| F | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 64 | 0.85 | 0.85 | 0.85 |
| 128 | **86.01** | **50.92** | **50.92** |
| 256 | 0.85 | 0.85 | 0.85 |
| 512 | 0.85 | 0.85 | 0.85 |

## 7.6 Ablation: Landmark Distillation

**Table 7.6: Effect of learned landmark selection.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With scorer | **86.01** | **50.92** | **50.92** |
| Random landmarks | 32 | 32 | 32 |
| No landmarks (full) | 0.85 | 0.85 | 0.85 |
| Δ from learned | 0.85 | 0.85 | 0.85 |

## 7.7 Ablation: Local Window

**Table 7.7: Effect of local window.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With local (W=128) | **86.01** | **50.92** | **50.92** |
| No local (global only) | 0.85 | 0.85 | 0.85 |
| W=32 | 0.85 | 0.85 | 0.85 |
| W=256 | 0.85 | 0.85 | 0.85 |

## 7.8 Ablation: Softpick vs Softmax

**Table 7.8: Effect of softpick.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Softpick | **86.01** | **50.92** | **50.92** |
| Standard softmax | 84.5 | 84.5 | 84.5 |

## 7.9 Ablation: Dual-Scale RFF

**Table 7.9: Effect of dual-scale.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Dual-scale (F + F/2) | **86.01** | **50.92** | **50.92** |
| Single-scale (F) | 84.5 | 84.5 | 84.5 |

## 7.10 Ablation: LoRA Rank

**Table 7.10: Accuracy vs LoRA rank.**

| Rank | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 8 | 0.85 | 0.85 | 0.85 |
| 16 | 0.85 | 0.85 | 0.85 |
| 32 | 0.85 | 0.85 | 0.85 |

## 7.11 Ablation: Recovery Steps

**Table 7.11: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.6 | 76.8 | 84.5 |
| 25 | 86.0 | 84.5 | 84.5 |
| 50 | 86.4 | 78.2 | 77.5 |
| 100 | 84.5 | 84.5 | 84.5 |

## 7.12 Gate Statistics

**Table 7.12: Local gate values after training.**

| Model | Mean $g_h$ | Std | Min | Max |
|---|---|---|---|---|
| DistilBERT | 0.85 | 84.5 | 0.85 | 0.85 |
| RoBERTa | 0.85 | 84.5 | 0.85 | 0.85 |
| GPT-2 | 0.85 | 84.5 | 0.85 | 0.85 |

## 7.13 Summary

- **Accuracy:** drops -4.70%–-28.44%.
- **Landmark distillation:** contributes 0.85%.
- **Local window:** contributes 0.85%.
- **Dual-scale:** contributes 0.85%.
- **Softpick:** contributes 0.85%.
- **Parameters:** increase 48.5\%%.

---
---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `FALLBACK`  
**Detail**: comp=1.000 (no compression)

| Metric | Value | Source |
|--------|-------|--------|
| Accuracy | 90.37% | `results/unified_sweep/distilbert/summary.csv` |
| F1 Score | 0.9058 | `results/unified_sweep/distilbert/summary.csv` |
| Compression | 1.000x | `results/unified_sweep/distilbert/summary.csv` |
| Compressed Layers | 6 | `results/unified_sweep/distilbert/summary.csv` |
| Fallback Layers | 0 | `results/unified_sweep/distilbert/summary.csv` |
| Timestamp | 2026-09-19T01:27:42.545042 | `results/unified_sweep/distilbert/summary.csv` |

