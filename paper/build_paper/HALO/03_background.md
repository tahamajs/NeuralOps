<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Dense Softmax Attention

For $Q, K, V \in \mathbb{R}^{L \times d_h}$:

$$
\mathcal{A}_\text{dense}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V \in \mathbb{R}^{L \times d_h}
\tag{3.1}
$$

**Complexity.** $\mathcal{O}(L^2 d_h)$ time, $\mathcal{O}(L^2)$ memory.

**Causality.** For decoder models, mask: $S_{ij} = -\infty$ for $j > i$.

## 3.2 Linear Attention

Replace softmax with a kernel $\phi: \mathbb{R}^{d_h} \to \mathbb{R}^m$:

$$
\mathcal{A}_\text{lin}(Q, K, V) = \frac{\phi(Q) \left(\phi(K)^\top V\right)}{\phi(Q) \left(\phi(K)^\top \mathbf{1}\right)}
\tag{3.2}
$$

where $\mathbf{1} \in \mathbb{R}^L$ is a vector of ones.

**Complexity.** $\mathcal{O}(L m d_h)$ time, $\mathcal{O}(m d_h)$ memory.

**Non-causal.** Direct computation via $\phi(K)^\top V \in \mathbb{R}^{m \times d_h}$.

**Causal.** Use prefix sums:
$$
\text{Num}_i = \phi(Q_i) \cdot \sum_{j \le i} \phi(K_j) V_j^\top, \quad
\text{Den}_i = \phi(Q_i) \cdot \sum_{j \le i} \phi(K_j)
\tag{3.3}
$$

## 3.3 Rectified Kernel

We use $\phi(x) = \mathrm{elu}(x) + 1$ where:

$$
\mathrm{elu}(x) = \begin{cases} x & x > 0 \\ \alpha (e^x - 1) & x \leq 0 \end{cases}
\tag{3.4}
$$

with $\alpha = 1$. This kernel:

- Is **non-negative**: $\phi(x) > 0$ for all $x$.
- Is **continuous** and **smooth**.
- Is **bounded** in expectation for bounded inputs.

**Why non-negativity matters.** In (3.2), the denominator 
$\phi(Q) (\phi(K)^\top \mathbf{1})$ is a sum of products of non-negative 
terms, hence positive. This avoids division by zero or negative denominators.

## 3.4 Prefix Sum for Causal Attention

For causal linear attention, we compute:

$$
\text{cum}_i = \sum_{j \le i} \phi(K_j) V_j^\top \in \mathbb{R}^{m \times d_h}
\tag{3.5}
$$

This is a **cumulative sum** over the sequence dimension, which we implement 
via `torch.cumsum`. The complexity is $\mathcal{O}(L m d_h)$ memory and time.

**Numerical stability.** Cumsum in FP32 avoids overflow for $L \le 8192$ 
with $m \le 128$. For longer sequences, chunked cumsum or a recurrent 
formulation is needed.

## 3.5 Gating

We use a per-head sigmoid gate:

$$
g_{h,i} = \sigma\!\left(w_h^\top x_i + b_h\right) \in (0, 1)
\tag{3.6}
$$

The output is a convex combination:

$$
O_i = g_{h,i} \cdot O_i^\text{dense} + (1 - g_{h,i}) \cdot O_i^\text{lin}
\tag{3.7}
$$

**Interpretation.** $g \approx 1$ → dense attention; $g \approx 0$ → linear 
attention; intermediate → smooth mix.

## 3.6 Grouped Query Attention (GQA)

GQA (Ainslie et al., 2023) reduces $H_{kv} < H$ and repeats KV heads:

$$
K, V \in \mathbb{R}^{B \times H_{kv} \times L \times d_h}, \quad
\tilde{K}, \tilde{V} = \text{repeat\_kv}(K, V, H / H_{kv})
\tag{3.8}
$$

HALO-ReGLA supports GQA via `repeat_kv` before attention computation.

## 3.7 Rotary Position Embeddings (RoPE)

RoPE (Su et al., 2021) applies position-dependent rotation:

$$
\tilde{Q}_i = Q_i \cos(\theta_i) + \mathrm{rot}(Q_i) \sin(\theta_i)
\tag{3.9}
$$

HALO-ReGLA applies RoPE to Q, K before both branches.

## 3.8 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r} B A
\tag{3.10}
$$

with $B = 0$ at initialization. HALO-ReGLA uses LoRA to adapt the frozen 
projections.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $\phi(\cdot)$ | Feature map |
| $\text{cum}_i$ | Prefix sum at position $i$ |
| $g_{h,i}$ | Gate value |
| $\sigma$ | Sigmoid |
| $\alpha$ | LoRA scaling |
| $r$ | LoRA rank |

---
