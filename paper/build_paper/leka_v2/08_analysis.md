# 8. Analysis

## 8.1 Contribution of Each Component

**Table 8.1: Ablation of components (DistilBERT).**

| Config | Accuracy | Δ |
|---|---|---|
| Baseline (linear attention only) | 0.85 | — |
| + Dual-scale RFF | 0.85 | +0.85 |
| + Landmark distillation | 0.85 | +0.85 |
| + Local window | 0.85 | +0.85 |
| + Softpick | **86.01** | +0.85 |

All four components contribute.

## 8.2 Landmark Selection Analysis

**Figure 8.1** shows the distribution of selected landmarks.

- **Position 0** is always retained (guaranteed).
- **Token 1-5** are frequently selected (early context).
- **Scattered positions** throughout the sequence based on content.

## 8.3 Why Softpick Matters

**Figure 8.2** shows the case where a full row of attention is masked. 
Standard softmax gives uniform attention (all entries $= 1/N$); 
softpick gives **all zeros** and falls back to uniform only when needed.

**Result.** Softpick prevents numerical instability at masked rows.

## 8.4 Local Window Contribution

**Table 8.2: Effect of window size.**

| W | DistilBERT | Efficiency |
|---|---|---|
| 32 | 0.85 | Fastest |
| 128 | 86.01 | Sweet spot |
| 256 | 0.85 | Slower |
| Full L | 0.85 | Same as MHSA |

## 8.5 Dual-Scale Analysis

The two RFF projections capture different frequency bands. **Figure 8.3** 
shows the singular value spectra of $W_1$ and $W_2$:

- $W_1$ (F=128): captures low-frequency kernel features.
- $W_2$ (F/2=64): captures high-frequency kernel features.

Combining them gives a richer representation.

## 8.6 Learned Local Gate

**Figure 8.4** shows per-head gate values after training. Average values:

| Model | Mean $g_h$ |
|---|---|
| DistilBERT | 0.85 |
| RoBERTa | 0.85 |
| GPT-2 | 0.85 |

Heads with high $g_h$ prefer global; low $g_h$ prefer local.

## 8.7 Complexity Reality Check

**Table 8.3: Actual complexity.**

| Component | Complexity |
|---|---|
| Local window | $O(L \cdot W)$ |
| Global landmark | $O(L \cdot L_\text{max})$ |
| **Total** | **$O(L \cdot (W + L_\text{max}))$** |

With $L_\text{max} = \rho L$ and $\rho = 0.5$, the landmark term is 
$O(L^2 / 2)$ — still quadratic but with half the constant.

**Note.** LEKAv2 is **not truly linear** unless $L_\text{max}$ is fixed.

## 8.8 Failure Modes

### 8.8.1 Fatal Error Handling

The `copy_weights` still raises `RuntimeError` on extractor failure.

### 8.8.2 Landmark Saturation

If the scorer assigns equal scores to all tokens, selection is effectively 
random.

### 8.8.3 Local Gate Saturation

If $g_h \to 0$ or $g_h \to 1$, one branch is ignored.

## 8.9 Summary of Insights

1. **All four components contribute.**
2. **Landmark distillation is essential.**
3. **Dual-scale helps.**
4. **Softpick stabilizes.**
5. **LEKAv2 is not truly linear at ρ = 0.5.**

---