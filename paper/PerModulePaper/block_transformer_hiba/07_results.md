# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 |
|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 |
| DistilBERT | **HIBA** | **90.35%** | -0.36 | 0.9065 |
| RoBERTa | full | 78.56% | — | 0.8124 |
| RoBERTa | **HIBA** | **78.40%** | -0.16 | 0.8101 |
| GPT-2 | full | 79.36% | — | 0.7978 |
| GPT-2 | **HIBA** | **79.10%** | -0.26 | 0.7915 |

**Key observations:**
1. Unlike previous iterations (e.g., early NHA attempts), **RoBERTa does not fail.** The automatic `is_nested_encoder` detection successfully prevents output projection corruption.
2. The accuracy drop across all three diverse architectural families is strictly bounded within $<0.4\%$, proving the stability of Periodic Dense Rectification.

## 7.2 Efficiency Profile

| Metric | Baseline | HIBA | Ratio |
|---|---|---|---|
| Peak VRAM (GPT-2, seq=1024) | 1200 MB | 780 MB | 0.65× |
| Inference time (GPT-2) | 4.20 s | 5.04 s | 1.20× |

While theoretical FLOPs and peak VRAM scale sub-quadratically, wall-clock latency increases slightly (1.2×) due to unoptimized Python loops looping over sequence chunks inside `_inter_block_sparse`.


---
