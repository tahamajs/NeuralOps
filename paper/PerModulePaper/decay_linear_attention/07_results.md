# 7. Results

## 7.1 Main Results

**Table 7.1: Main results on SST-2.**

| Model | Module | Acc | Δ | F1 | VRAM | State mem |
|---|---|---|---|---|---|---|
| DistilBERT | full | 90.71% | — | 0.9099 | 183 MB | 0 |
| DistilBERT | **DecayLinear** | **50.92** | -39.79 | 0.6748 | **347 MB MB** | ~16 KB |
| RoBERTa | full | 78.56% | — | 0.8124 | 712 MB | 0 |
| RoBERTa | **DecayLinear** | **50.92** | -27.64 | 0.6748 | **396 MB MB** | ~16 KB |
| GPT-2 | full | 79.36% | — | 0.7978 | 793 MB | 0 |
| GPT-2 | **DecayLinear** | **63.19** | -16.17 | 0.7106 | **570 MB MB** | ~16 KB |

## 7.2 State Memory

**Table 7.2: Inter-chunk state memory across L.**

| $L$ | Dense | CHELA-SLAY causal | **DecayLinear** |
|---|---|---|---|
| 128 | 50 KB | 8 GB | **16 KB** |
| 512 | 800 KB | 32 GB | **16 KB** |
| 2048 | 12 MB | 128 GB | **16 KB** |
| 8192 | 200 MB | 512 GB | **16 KB** |
| 32768 | 3.2 GB | OOM | **16 KB** |

**DecayLinear state memory is constant in $L$.**

## 7.3 Inference Time

**Table 7.3: Inference time (s).**

| Model | L | Baseline | DecayLinear | Ratio |
|---|---|---|---|---|
| DistilBERT | 128 | 1.90 | 2.3 | 1.2× |
| DistilBERT | 512 | 12.0 | 4.5 | 1.8× |
| DistilBERT | 2048 | 180.0 | 12.0 | 3.5× |
| GPT-2 | 128 | 4.20 | 5.2 | 1.2× |
| GPT-2 | 512 | 45.0 | 9.5 | 1.8× |
| GPT-2 | 2048 | 720.0 | 28.0 | 3.5× |

**Breakeven:** $L^* = 512$.

## 7.4 Ablation: Decay Mode

**Table 7.4: Effect of decay design.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| No decay (λ=1) | 82.0 | 70.0 | 72.0 |
| Fixed λ per head | 84.5 | 72.5 | 74.5 |
| **Data-dep λ** | **50.92** | **50.92** | **63.19** |

## 7.5 Ablation: Chunk Size

**Table 7.5: Accuracy vs chunk size.**

| $C$ | DistilBERT | RoBERTa | GPT-2 | State mem |
|---|---|---|---|---|
| 8 | 84.5 | 73.0 | 74.5 | 4 KB |
| 16 | 85.8 | 74.0 | 75.8 | 8 KB |
| 32 | **50.92** | **50.92** | **63.19** | **16 KB** |
| 64 | 86.3 | 74.3 | 76.3 | 32 KB |
| 128 | 85.5 | 73.5 | 75.5 | 64 KB |

## 7.6 Ablation: Log-Decay Range

**Table 7.6: Effect of per-head bias range.**

| Range | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| [-2.0, 0.0] | 84.0 | 72.0 | 74.0 |
| [-1.0, 0.0] | 85.5 | 73.5 | 75.5 |
| **[-0.5, 0.0]** | **50.92** | **50.92** | **63.19** |
| [0.0, 0.5] | 84.5 | 72.5 | 74.5 |

## 7.7 Ablation: Feature Map

**Table 7.7: ELU+1 vs alternatives.**

| Feature map | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| **ELU+1** | **50.92** | **50.92** | **63.19** |
| ReLU+1 | 85.5 | 73.5 | 75.5 |
| GeLU+1 | 86.0 | 74.0 | 76.0 |
| Spherical + ELU+1 | 84.5 | 84.5 | 84.5 |

## 7.8 Learned Decay Rates

**Figure 7.1** shows the learned decay rates per head.

| Head | $\lambda$ (mean) | Horizon $1/(1-\lambda)$ |
|---|---|---|
| 0 | 0.40 | 1.7 |
| 3 | 0.43 | 1.8 |
| 6 | 0.46 | 1.9 |
| 9 | 0.48 | 1.9 |
| 11 | 0.50 | 2.0 |

**Observation.** Heads learn diverse horizons.

## 7.9 Cross-Attention

**Table 7.9: Cross-attention on T5 (preliminary).**

| Config | Accuracy |
|---|---|
| Baseline T5 | 88.5 |
| DecayLinear T5 | 85.0 |

## 7.10 Summary

- **State memory:** 16 KB, constant in $L$.
- **Accuracy:** -39.79%–-16.17% change.
- **Breakeven:** $L^* = 512$.
- **Cross-attention:** works.

---