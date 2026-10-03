<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Low-Rank Compression

**SVD factorization** (Eckart & Young, 1936). The optimal rank-$r$ 
approximation in Frobenius norm.

**ALBERT** (Lan et al., 2020). Cross-layer parameter sharing.

**Tensor train** (Novikov et al., 2015). Higher-order factorization.

**ORION uses SVD** with energy-based rank selection.

## 2.2 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020). $\phi(x) = \text{elu}(x) + 1$.

**Performer** (Choromanski et al., 2021). Random Fourier Features.

**Cosformer** (Qin et al., 2022). Cosine reweighting.

**ORION uses RFF** for the kernel heads.

## 2.3 Hybrid Attention

**Longformer** (Beltagy et al., 2020). Sliding-window + global.

**BigBird** (Zaheer et al., 2020). Sliding-window + random + global.

**NASC** (this framework). Three-branch cascade.

**ORION bifurcates by head**, not by branch — different heads use different 
mechanisms.

## 2.4 Softpick

**Alman & Song (2024).** Sparse alternative to softmax.

**NASC's softpick** uses ReLU with fallback.

**ORION's softpick** uses $\text{ReLU}(e^x - 1)$, simpler.

## 2.5 Position of Our Work

| Axis | Prior work | ORION |
|---|---|---|
| **Factorization** | Post-hoc SVD | Zero-shot SVD |
| **Head mechanism** | Uniform | Bifurcated (2/3, 1/3) |
| **Parameter reduction** | Often none | **~6×** |
| **Robustness** | Varies | **`copy_weights` raises** |

**Key distinction.** ORION is one of the few modules in our framework 
that **actually reduces parameters**. It combines three orthogonal 
techniques, though one of them (the `copy_weights` raise) is a bug.

---

## Comparison Table

| Method | SVD? | Head split? | RFF? | Params Δ |
|---|---|---|---|---|
| Baseline | No | No | No | 0 |
| NHA | No | No | No | +0.88% |
| Cosformer | No | No | No | +5.7% |
| A3 | Yes | No | No | **-60%** |
| **ORION** | **Yes** | **Yes** | **Yes** | **-32%** |

---