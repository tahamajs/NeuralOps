<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Low-rank factorization is a standard approach for compressing transformer 
attention. Each weight matrix $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ 
is replaced with:

$$
W \approx U V, \qquad
U \in \mathbb{R}^{d_\text{out} \times r}, \quad
V \in \mathbb{R}^{r \times d_\text{in}}
\tag{1.1}
$$

reducing parameters from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$. 
Classical SVD gives the optimal approximation at a given rank.

**The rank-selection problem.** Rank $r$ is a **discrete hyperparameter**. 
Its choice presents a hard trade-off:

- **Too small** — the approximation error is large, and recovery cannot 
  compensate.
- **Too large** — the compression is minimal, defeating the purpose.

Existing approaches handle this in one of three ways:

1. **Fixed rank.** Simple but suboptimal — different layers have different 
   intrinsic ranks.
2. **Energy threshold.** Choose the smallest $r$ with cumulative spectral 
   energy $E(r) \geq \eta$. Training-free but the threshold $\eta$ is 
   itself a hyperparameter.
3. **Post-hoc search.** Grid search rank per layer. Expensive and 
   discontinuous.

All three treat rank as a **discrete** quantity that must be chosen before 
or after training, not **during** it.

## 1.2 Problem Statement

We seek a low-rank factorization $W \approx U V$ where the rank is 
**learnable end-to-end** with the recovery fine-tuning, so that:

1. Each layer's effective rank is chosen by gradient descent.
2. The initial forward pass equals the truncated SVD approximation 
   (no accuracy loss at step 0).
3. After training, ranks that turned out to be unnecessary can be pruned 
   without retraining.

Formally: we replace the hard constraint $\text{rank}(W) = r$ with a 
**soft, differentiable** parameterization.

## 1.3 Our Approach: Dobi-SVD

**Dobi-SVD** (Differentiable SVD) introduces a learnable **spectral gate** 
$\alpha \in \mathbb{R}^r$ that scales each rank component:

$$
y = (x (V \odot \sigma(\alpha))^\top) U^\top + b
\tag{1.2}
$$

where $\sigma$ is the sigmoid function. Each gate $\alpha_i$ controls the 
contribution of the $i$-th singular direction.

**Three design choices:**

1. **Gate initialization at $\alpha_i = 0.5$.** This gives 
   $\sigma(\alpha_i) \approx 1.0$, so the model starts near 
   the truncated SVD.

2. **Gate-compensation pre-scaling of $V$.** We set:

   $$
   V \leftarrow V / \sigma(\alpha_\text{init})
   \tag{1.3}
   $$

   so that at step 0, the forward pass is **exactly** the truncated SVD. 
   The gating is a refinement, not a perturbation.

3. **Learnable gating.** During recovery training, gates co-adapt with LoRA 
   adapters. At the end of training, gates near zero identify components 
   that can be pruned.

## 1.4 Contributions

- **Differentiable rank selection.** A learnable spectral gate that makes 
  rank a continuous parameter, trainable end-to-end with LoRA recovery.
- **Near-lossless initialization.** Gate compensation ensures that step 0 
  matches the truncated SVD exactly.
- **Post-hoc pruning without retraining.** After training, gates below a 
  threshold identify removable components.
- **Multi-architecture hardening.** Supports flat, nested, causal Conv1D, 
  and GQA models with LoRA wrapping preservation.
- **Full reproducibility.** Code, weights, and scripts released under 
  Apache 2.0.

## 1.5 Paper Organization

Section 2 reviews related work on rank selection. Section 3 provides 
background. Section 4 presents Dobi-SVD. Section 5 describes architecture 
integration. Section 6 details setup. Section 7 reports results. Section 8 
analyzes gate distributions. Section 9 discusses guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ | Weight matrix |
| $U \in \mathbb{R}^{d_\text{out} \times r}$ | Left factor |
| $V \in \mathbb{R}^{r \times d_\text{in}}$ | Right factor (pre-gating) |
| $\alpha \in \mathbb{R}^r$ | Learnable gate logits |
| $\sigma(\alpha)$ | Sigmoid-applied gates |
| $r$ | Nominal rank |
| $r_\text{eff}$ | Effective rank (post-training) |
| $\tau_g$ | Gate pruning threshold |
| $d, H, H_{kv}, d_h$ | Model dims |
| $\eta$ | Spectral energy threshold (for initialization) |

---