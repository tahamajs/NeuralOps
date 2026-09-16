# 8. Analysis

## 8.1 Why Does PADRe Work?

### 8.1.1 Taylor Approximation Quality

**Figure 8.1** compares the polynomial kernel to softmax for various orders. 
Order 2 matches softmax well for $|s| < 1$; order 4 for $|s| < 2$.

After centering, most scores fall in $[-2, 0]$, so order 4 is sufficient.

### 8.1.2 Centering Prevents Overflow

**Figure 8.2** shows the base values with and without centering. Without 
centering, scores can reach 20+, causing overflow in FP16.

### 8.1.3 ReLU Sparsity

**Figure 8.3** shows the fraction of zeroed positions. After centering, 
positions with $s < -d$ get zero weight. This creates natural sparsity.

## 8.2 Order Analysis

**Table 8.1: Taylor error vs order.**

| Order | Max error ($|s| \leq 1$) | Max error ($|s| \leq 2$) |
|---|---|---|---|
| 1 | 0.20 | 0.80 |
| 2 | 0.08 | 0.40 |
| 4 | 0.01 | 0.10 |
| 8 | 0.001 | 0.01 |

## 8.3 Comparison with Softmax

**Table 8.2: Numerical differences.**

| Metric | Softmax | PADRe (order 4) |
|---|---|---|
| Max weight | 0.95 | 0.93 |
| Min weight (nonzero) | 0.001 | 0.005 |
| Sparsity | 0% | 15% |

## 8.4 Injection Behavior

**Figure 8.4** shows the output equivalence error for each layer. Layers 
with high error (e.g., early layers with peaked attention) are skipped.

## 8.5 Numerical Stability

**Table 8.3: Failure modes and mitigations.**

| Failure | Mitigation |
|---|---|
| Overflow in base | Max-centering |
| Negative base NaN | ReLU clamp |
| All-masked query | Set max to 0 |
| FP16 underflow | Use FP32 for scores |

## 8.6 Summary of Insights

1. **Order 4 is sufficient** for typical score ranges.
2. **Centering is essential** for stability.
3. **ReLU sparsity is a free benefit** (15% zero weights).
4. **Injection check prevents bad layers.**
5. **Runtime is unchanged** without kernelization.

---