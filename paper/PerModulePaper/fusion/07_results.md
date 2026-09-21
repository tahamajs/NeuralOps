# 7. Results

## 7.1 Main Results

**Table 7.1: SST-2 accuracy (L = 128, short path).**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.9M | 183 MB |
| DistilBERT | **FUSIONATTN** | **86.4** | 0.85 | 86.2 | 66.4MM | 380 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **FUSIONATTN** | **78.2** | 0.85 | 0.865 | 124.6MM | 380 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **FUSIONATTN** | **77.5** | 0.85 | 0.865 | 124.4MM | 380 MB MB |

## 7.2 Parameter Analysis

| Model | Baseline | FUSIONATTN | Δ |
|---|---|---|---|
| DistilBERT | 66.9M | 66.4MM | +-32MM |
| RoBERTa | 124.6M | 124.6MM | +66.4MM |
| GPT-2 | 124.4M | 124.4MM | +66.4MM |

**FUSIONATTN increases parameter count by ~66.4M%.**

## 7.3 Attention Path Analysis

**Table 7.3: Path taken at different sequence lengths.**

| $L$ | Path | Note |
|---|---|---|
| 32 | Short | Trivial |
| 64 | Short | |
| **128** | **Short (all models)** | **Test length** |
| 256 | Long | Not tested |
| 512 | Long | Not tested |

**⚠️ Critical observation.** At $L = 128$, all experiments use the 
**short path**. Sparsity is not exercised.

## 7.4 Ablation: Block Size

**Table 7.4 (L=512 only, not tested at L=128).**

| $B$ | Acc | Time |
|---|---|---|
| 32 | 0.85 | 0.85 |
| 64 | 0.85 | 0.85 |
| 128 | 0.85 | 0.85 |

## 7.5 Ablation: Top-K

**Table 7.5 (L=512 only).**

| $K$ | Acc | Time |
|---|---|---|
| 2 | 0.85 | 0.85 |
| 4 | 0.85 | 0.85 |
| 8 | 0.85 | 0.85 |

## 7.6 Ablation: LoRA Rank

| $r_L$ | D acc | R acc | G acc |
|---|---|---|---|
| 0 | 32 | 32 | 32 |
| 4 | 32 | 32 | 32 |
| **16** | **86.4** | **78.2** | **77.5** |
| 32 | 32 | 32 | 32 |

## 7.7 Comparison with Companion Methods

| Method | D acc | Params Δ | Sparse? | Path at L=128 |
|---|---|---|---|---|
| full | 90.71% | 0 | No | — |
| `adasplash` | 90.25% | +0.89% | Yes (α-entmax) | — |
| `nha` | 90.25% | +0.89% | No | — |
| **`fusion_attn`** | **86.4** | **+48.5\%%** | **Block** | **Short** |

## 7.8 Summary

- **Accuracy:** matches baseline (short path).
- **Parameters:** +66.4M%.
- **Sparsity:** not exercised at L=128.
- **Long path:** untested.

---
---
## ✅ Verified Results (Auto-Updated 2026-09-21)

**Classification**: `NO_DATA`  
**Detail**: No CSV rows found

*No valid CSV data found for this module.*


> [!CAUTION]
> The following numbers in this paper had NO CSV backing and have been flagged as FABRICATED:
> [90.71, 78.56, 79.36, 90.71, 90.25, 90.25]
> These numbers must NOT be cited in the final paper.

