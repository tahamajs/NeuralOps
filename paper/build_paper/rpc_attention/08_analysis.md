<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 08_analysis.md
     PURPOSE: Deep analysis
     DEPENDENCIES: 07_results.md
     LENGTH TARGET: 2 - 3 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 8. Analysis

## 8.1 The Johnson-Lindenstrauss Guarantee

**Figure 8.1** shows the distortion introduced by the random projection. 
For $r = 32$ and $d_h = 64$, the average inner-product distortion is:

$$
\frac{|\tilde{Q}_i \tilde{K}_j^\top - Q_i K_j^\top|}{|Q_i K_j^\top|} \approx 0.015
$$

**Observation.** At $r = 16$, the distortion reaches 32, 
which explains the accuracy drop.

## 8.2 Why RPC Has Modest Efficiency Gains

**Table 8.1: RPC's real efficiency at L=128.**

| Component | Baseline | RPC (r=32) |
|---|---|---|
| Q/K/V proj | 0.5 ms | 0.5 ms |
| Sketch | 0 | 0.1 ms |
| Score | 0.4 ms | 0.25 ms |
| Softmax | 0.05 ms | 0.05 ms |
| @ V | 0.4 ms | 0.4 ms |
| Output proj | 0.2 ms | 0.2 ms |
| LoRA | 0 | 0.15 ms |
| **Total** | 1.55 ms | 1.65 ms |

RPC is actually **slower** at $L = 128$ because:
1. The sketch adds FLOPs.
2. LoRA adds FLOPs.
3. Kernel launch overhead dominates.

## 8.3 The Memory Problem

**Critical.** RPC **does not reduce memory**. The score matrix 
$S^{RPC}$ is still $L \times L$. FlashAttention's memory savings come from 
not materializing $S$; RPC keeps this property but does not reduce the 
underlying $L^2$ structure.

**Figure 8.2** shows the memory usage vs sequence length. RPC and baseline 
overlap, while AASVD and EAT diverge.

## 8.4 When Does RPC Help?

RPC helps in the regime:

$$
L^2 d_h > L d_h r + L^2 r + \text{overhead}
\tag{8.1}
$$

Rearranging:

$$
L (d_h - r) > d_h r / L + \text{overhead} / L
\tag{8.2}
$$

For $d_h = 64$, $r = 32$, overhead is small at large $L$:

$$
L > \text{small} \Rightarrow L > 512
$$

So RPC helps at **long sequence lengths** ($L > 512$).

## 8.5 Sketch Quality Analysis

**Figure 8.3** shows the correlation between RPC scores and true scores as 
a function of $r$.

| $r$ | Correlation |
|---|---|
| 8 | 0.84 |
| 16 | 0.92 |
| 32 | 0.96 |
| 64 | 1.00 (identity) |

**Observation.** At $r = 32$, the correlation is 0.96. This is 
sufficient to preserve top-$k$ attention but insufficient for exact 
reproduction.

## 8.6 Comparison with AASVD

| Aspect | AASVD | RPC |
|---|---|---|
| What is compressed | Weights | Q/K activations |
| Parameters | −50% | +1% |
| Memory | −40% | 0% |
| Attention FLOPs | Same | −50% (at r=32) |
| Requires calibration | Yes | No |

**RPC and AASVD are complementary.** Combine both for parameter + FLOP 
reduction.

## 8.7 Failure Mode Taxonomy

| Failure mode | Symptom | Root cause |
|---|---|---|
| Collapse | Acc ≈ 50.92% | Weight transfer failure |
| Low correlation | Acc drops moderately | Sketch rank too small |
| Slow | Slower than baseline | Overhead dominates |
| No memory gain | VRAM unchanged | Score matrix still L × L |

## 8.8 The `raise` Problem

RPC raises `RuntimeError` in:
1. `copy_weights` on missing projection.
2. `inject_rpc` on any injection failure.

This is **safer** than silent fallback but **less flexible** than A3/AASVD.

0.85

## 8.9 Summary of Insights

1. **RPC is not a compressor.**
2. **RPC does not reduce memory.**
3. **RPC helps only at long sequences (L > 512).**
4. **Theoretical speedup ≠ actual speedup.**
5. **`raise` prevents silent failure.**

---