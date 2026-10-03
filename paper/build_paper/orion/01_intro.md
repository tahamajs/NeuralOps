<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The transformer's parameter count is dominated by four weight matrices 
per layer: $W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ for self-attention, 
plus two FFN projections. For a $d = 768$, 12-layer model, this amounts 
to ~85M parameters — most of the model.

Two orthogonal strategies can reduce this:

1. **Low-rank factorization:** Replace each $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
   with a product $W_\text{up} W_\text{down}$ where $W_\text{down} \in \mathbb{R}^{r \times d_\text{in}}$ 
   and $W_\text{up} \in \mathbb{R}^{d_\text{out} \times r}$. The parameter 
   count drops from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$.

2. **Efficient attention:** Replace the $\mathcal{O}(L^2)$ softmax attention 
   with a linear or sparse alternative.

Most methods address one or the other. **ORION combines both.**

## 1.2 Our Approach

ORION has three distinct components:

### 1.2.1 Zero-Shot SVD Factorization

Each Q/K/V/O projection is factorized via SVD at rank $r = 64$, with 
the top-$r$ singular vectors retained:

$$
W \approx U_r \Sigma_r V_r^\top = (U_r \Sigma_r^{1/2})(\Sigma_r^{1/2} V_r^\top)
$$

The factors are frozen; LoRA adapters provide recovery.

### 1.2.2 Head-Wise Bifurcation

Attention heads are split into two groups:

- **Kernel heads** (2/3): linear attention via random Fourier features.
- **Full heads** (1/3): full softmax-equivalent attention via softpick.

This is motivated by the observation that some heads learn low-rank 
patterns (kernel-friendly), while others require full attention.

### 1.2.3 Random Fourier Features

For the kernel heads:

$$
\phi(x) = \frac{1}{\sqrt{2F}} [\cos(\Omega x), \sin(\Omega x)]
$$

with $\Omega \in \mathbb{R}^{d_h \times F}$ orthogonally initialized.

## 1.3 Contributions

- **A tri-component hybrid attention** combining SVD factorization, 
  head-wise bifurcation, and RFF.
- **Genuine parameter reduction** (~6× for square projections, ~9.6× for 
  FFN-like projections).
- **An honest disclosure** of the `copy_weights` raise-bug that can cause 
  catastrophic collapse.
- **Full reproducibility** including code and checkpoints.

## 1.4 A Critical Bug Disclosure

The `copy_weights` method contains:

```python
except Exception as e:
    print(f'copy_weights failed: {e}')
    raise RuntimeError(f"copy_weights failed: {e}")
```

If `UniversalAdapter.get_weight_bias` fails for any of the four 
projections (which can happen for architectures it does not recognize), 
the entire injection aborts with a `RuntimeError`. This matches the 
pattern observed in other modules (AEGIS V4/V5, EAT) that caused 
catastrophic accuracy collapse.

**This is a bug, not a feature.** We document it here and in Section 8 
for reproducibility.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $d$ | Hidden size | §1.1 |
| $L$ | Sequence length | §1.1 |
| $r$ | SVD rank | §1.2 |
| $H_k$ | Kernel heads | §1.2 |
| $H_f$ | Full heads | §1.2 |
| $F$ | RFF features | §1.2 |
| $\Omega$ | RFF projection | §1.2 |
| $r_L$ | LoRA rank | §1.2 |

---