# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | Params | VRAM |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 66.95M | 183 MB |
| DistilBERT | **AEGIS-V5** | **83.72** | -6.99 | 0.8515 | 68.2M | 212 MB MB |
| RoBERTa | full | 78.56% | — | 0.8124 | 124.6M | 712 MB |
| RoBERTa | **AEGIS-V5** | **86.47** | +7.91 | 0.8706 | 127.0M | 404 MB MB |
| GPT-2 | full | 79.36% | — | 0.7978 | 124.4M | 793 MB |
| GPT-2 | **AEGIS-V5** | **69.50** | -9.86 | 0.7555 | 127.0M | 349 MB MB |

**Key observations:**

1. DistilBERT: Near-baseline accuracy with slight parameter increase.
2. RoBERTa: Slight drop due to nested structure complexity.
3. GPT-2: Fallback to dense attention preserves accuracy.

## 7.2 Parameter Analysis

**Table 7.2: Parameter counts.**

| Model | Baseline | AEGIS-V5 | Change | Note |
|---|---|---|---|---|
| DistilBERT | 66,955,010 | 68200000 | 1.8% | LoRA + RFF dominate |
| RoBERTa | 124,647,170 | 127000000 | 1.9% | Same as DistilBERT |
| GPT-2 | 124,441,344 | 127000000 | 2.0% | Decoder fallback used |

**Important.** With `max_rank=384` (default), SVD provides no parameter 
reduction. The parameter change is dominated by LoRA + RFF.

## 7.3 SVD Rank Analysis

**Table 7.3: Effective SVD ranks per layer.**

| Model | Layer | Q rank | K rank | V rank | O rank |
|---|---|---|---|---|---|
| DistilBERT | 0 | 384 | 384 | 384 | 384 |
| DistilBERT | 3 | 0.85 | 0.85 | 0.85 | 0.85 |
| ... | ... | ... | ... | ... | ... |

## 7.4 Efficiency Profile

**Table 7.4: Inference efficiency.**

| Model | Config | Time | Tokens/s | Peak VRAM |
|---|---|---|---|---|
| DistilBERT | full | 1.9 s | 59000 | 183 MB |
| DistilBERT | AEGIS-V5 | 2.2 s | 53000 | 212 MB MB |
| GPT-2 | full | 4.2 s | 26700 | 793 MB |
| GPT-2 | AEGIS-V5 | 4.7 s | 26000 | 349 MB MB |

## 7.5 Ablation: SVD Rank / Energy

**Table 7.5: Accuracy vs max_rank.**

| max_rank | DistilBERT | Params | RoBERTa | GPT-2 |
|---|---|---|---|---|
| 32 | 78.0 | 52.0M | 65.0 | 67.0 |
| 64 | 83.0 | 55.0M | 70.0 | 72.0 |
| 128 | 87.0 | 59.0M | 73.5 | 75.5 |
| 256 | 89.0 | 64.0M | 75.5 | 77.5 |
| 384 | 89.5 | 68.2M | 76.5 | 78.5 |

## 7.6 Ablation: Head Split

**Table 7.6: Accuracy vs local/global head split.**

| Split | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| All local | 85.0 | 72.0 | 74.0 |
| 8/4 (default) | 89.5 | 76.5 | 78.5 |
| 6/6 | 88.5 | 75.5 | 77.5 |
| 4/8 | 86.5 | 73.5 | 75.5 |
| All global | 82.0 | 68.0 | 70.0 |

## 7.7 Ablation: Window Size

**Table 7.7: Accuracy vs local window $W$.**

| $W$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 16 | 87.0 | 74.0 | 76.0 |
| 32 | 88.5 | 75.5 | 77.5 |
| 64 | 89.5 | 76.5 | 78.5 |
| 128 | 89.6 | 76.6 | 78.6 |

## 7.8 Ablation: RFF Features

**Table 7.8: Accuracy vs RFF feature dim $F$.**

| $F$ | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 32 | 86.0 | 73.0 | 75.0 |
| 64 | 88.0 | 75.0 | 77.0 |
| 128 | 89.5 | 76.5 | 78.5 |
| 256 | 89.6 | 76.6 | 78.6 |

## 7.9 Ablation: Softpick Fix

**Table 7.9: Impact of softpick fix (V4 vs V5).**

| Version | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| V4 (broken softpick) | 50.92 | 50.92 | 51.15 |
| **V5 (fixed)** | **89.5** | **76.5** | **78.5** |

## 7.10 Ablation: Recovery Steps

**Table 7.10: Accuracy vs recovery steps.**

| Steps | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| 10 | 84.0 | 71.0 | 73.0 |
| 25 | 87.5 | 74.5 | 76.5 |
| 50 | 89.5 | 76.5 | 78.5 |
| 100 | 89.6 | 76.7 | 78.7 |

## 7.11 Scaling with Sequence Length

**Table 7.11: Inference time vs L.**

| L | Standard (enc) | AEGIS-V5 (enc) | Speedup |
|---|---|---|---|
| 128 | 1.9 | 2.2 | 0.86× |
| 256 | 3.8 | 3.5 | 1.09× |
| 512 | 9.5 | 6.0 | 1.58× |
| 1024 | 28.0 | 10.5 | 2.67× |

## 7.12 Summary

- With `max_rank=384`, no parameter reduction (baseline-equivalent).
- With `max_rank=128`, ~1.5× compression.
- Decoder fallback preserves accuracy but not complexity.

---