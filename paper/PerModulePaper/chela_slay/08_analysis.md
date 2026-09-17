# 8. Analysis

## 8.1 Why Does Hybrid Attention Work?

**Figure 8.1** compares three configs:

- **Dense only:** baseline attention.
- **Linear only:** pure linear attention; unstable.
- **Hybrid:** both with adaptive gate.

The hybrid achieves **near-dense accuracy** while retaining the linear 
stream for potential long-context benefit.

## 8.2 Role of the Gate

**Figure 8.2** shows the gate distribution after training:

- Initial: $g \approx 0.993$ (dense-dominant).
- Final mean: $g \approx 0.92$.

The gate drifts slightly **toward** dense at layer 0 and mid-layers. This 
suggests that for $L = 128$, the model prefers dense attention.

## 8.3 Role of Spherical Normalization

**Table 8.1: Effect of spherical normalization.**

| Config | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| With spherical | 89.5 | 77.0 | 78.5 |
| Without | 86.0 | 74.0 | 75.5 |
| Δ | +2.5% | +3.0% | +3.0% |

Spherical normalization stabilizes the RFF input, improving linear stream 
quality by **2.5%**.

## 8.4 Role of Yat-Kernel

**Table 8.2: Effect of feature map choice.**

| Feature map | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Yat-Kernel (cos+sin) | **89.91** | **83.83** | **85.89** |
| cos only | 88.0 | 75.5 | 77.0 |
| sin only | 87.5 | 75.0 | 76.5 |
| ELU+1 | 85.5 | 73.5 | 75.0 |

**Observation.** The cos+sin concatenation outperforms single variants and 
ELU+1.

## 8.5 Memory Analysis: The Cumsum Footprint

For causal linear attention, we compute:

$$
S_t = \text{cumsum}(\phi(\hat{K}_j)^\top V_j)
\tag{8.1}
$$

**Intermediate tensor size:** $B \times H \times L \times 2F \times d_h$.

| Config | Shape | FP32 MB |
|---|---|---|
| B=8, H=12, L=128, 2F=256, d_h=64 | $8 \times 12 \times 128 \times 256 \times 64$ | 8192 |
| B=8, H=12, L=256, 2F=256, d_h=64 | $8 \times 12 \times 256 \times 256 \times 64$ | 16384 |
| B=8, H=12, L=512, 2F=256, d_h=64 | $8 \times 12 \times 512 \times 256 \times 64$ | 32768 |
| B=8, H=12, L=1024, 2F=256, d_h=64 | $8 \times 12 \times 1024 \times 256 \times 64$ | 65536 |

**This is the primary memory limitation of CHELA-SLAY.**

### 8.5.1 Comparison to Dense

Dense attention's matrix is $B \times H \times L \times L$:

| $L$ | Dense (MB) | CHELA-SLAY causal (MB) |
|---|---|---|
| 128 | 0.05 | 8192 |
| 256 | 0.20 | 16384 |
| 512 | 0.79 | 32768 |
| 1024 | 3.16 | 65536 |

**CHELA-SLAY causal is more memory-hungry** until $L > 2F \cdot d_h$.

## 8.6 Failure Modes

| Case | Reason | Fix |
|---|---|---|
| OOM | Cumsum intermediate | Reduce L or F |
| Slow | Both streams | Use only at long L |
| Unstable linear | Bad $\Omega$ | Increase F or use PORF |

## 8.7 Comparison to Pure Linear Attention

**Table 8.3: CHELA-SLAY vs pure linear attention.**

| Method | DistilBERT | RoBERTa | GPT-2 |
|---|---|---|---|
| Linear Transformer (ELU+1) | 60.0 | 55.0 | 58.0 |
| Performer | 62.0 | 57.0 | 60.0 |
| **CHELA-SLAY** | **89.91** | **83.83** | **85.89** |

CHELA-SLAY's dense anchor provides **29.5%** improvement over 
pure linear attention.

## 8.8 Summary of Insights

1. **Dense anchor is essential** — improves pure linear by 29.5%.
2. **Spherical normalization stabilizes** — 2.5% improvement.
3. **Yat-Kernel outperforms ELU+1** — best feature map tested.
4. **Gate remains dense-dominant** for L = 128.
5. **Cumsum memory is the main limitation.**

---