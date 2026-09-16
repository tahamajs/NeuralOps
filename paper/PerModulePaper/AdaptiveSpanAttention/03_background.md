<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Softmax Attention

For a single head with $Q, K, V \in \mathbb{R}^{L \times d_h}$:

$$
S = \frac{Q K^\top}{\sqrt{d_h}} \in \mathbb{R}^{L \times L}
\tag{3.1}
$$

$$
P = \text{softmax}(S) \in \mathbb{R}^{L \times L}
\tag{3.2}
$$

$$
O = P V \in \mathbb{R}^{L \times d_h}
\tag{3.3}
$$

**Complexity.** $\mathcal{O}(L^2)$ memory and time.

## 3.2 Window Attention

**Fixed window.** Restrict attention to positions within distance $w$:

$$
P_{ij} = 0 \quad \text{if } |i - j| > w
\tag{3.4}
$$

For a causal model, $P_{ij} = 0$ for $j > i$ or $i - j > w$.

**Complexity.** $\mathcal{O}(L w)$ memory and time.

**Implementation.** Often done with band masks or block-sparse patterns.

## 3.3 Learned Span (Sukhbaatar et al., 2019)

**Idea.** Replace the fixed window $w$ with a per-head learned span $z_h$.

**Soft ramp.** Instead of hard cutoff, use a **continuous ramp**:

$$
g_z(d) = \text{clamp}\!\left(\frac{R + z - d}{R}, 0, 1\right)
\tag{3.5}
$$

- For $d \leq z$: $g_z(d) = 1$ (fully attend)
- For $z < d < z + R$: $g_z(d) = \frac{R + z - d}{R}$ (linear decay)
- For $d \geq z + R$: $g_z(d) = 0$ (no attend)

**Parameterization.** We use a scalar logit:

$$
z = \sigma(\ell) \cdot S_\text{max}
\tag{3.6}
$$

## 3.4 Span Regularization

**L1 penalty.** To encourage small spans:

$$
\mathcal{L}_\text{span} = \lambda \sum_{h=1}^H z_h
\tag{3.7}
$$

At initialization, $z_h \approx S_\text{max}$ and the penalty is large. During 
training, the model shrinks $z_h$ to reduce the penalty, subject to preserving 
task accuracy. The equilibrium depends on $\lambda$: larger $\lambda$ → 
smaller spans.

## 3.5 Softmax Renormalization

After applying the ramp mask:

$$
\tilde{P}_{ij} = \frac{P_{ij} \cdot g_{z_h}(|i - j|)}{\sum_j P_{ij} \cdot g_{z_h}(|i - j|)}
\tag{3.8}
$$

This renormalization ensures rows sum to 1. We implement this via:

$$
\tilde{P} = \text{softmax}(S + \log(g_z))
\tag{3.9}
$$

where we work in log space to avoid numerical issues.

## 3.6 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable low-rank adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r_L} B A
\tag{3.10}
$$

with $B = 0$ at initialization. ASA-Span uses LoRA to adapt the frozen 
projections to the new span structure.

## 3.7 Relative Position Bias

T5 (Raffel et al., 2020) adds a learned bias based on bucketed relative 
position:

$$
S_{ij} \leftarrow S_{ij} + b_{\text{bucket}(i - j)}
\tag{3.11}
$$

ASA-Span preserves this bias, applying the ramp **after** adding the bias.

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $z_h$ | Learned span for head $h$ |
| $\ell_h$ | Logit for head $h$ |
| $R$ | Ramp size |
| $g_z$ | Soft ramp mask |
| $\lambda$ | Span penalty weight |
| $S_\text{max}$ | Maximum span |

---