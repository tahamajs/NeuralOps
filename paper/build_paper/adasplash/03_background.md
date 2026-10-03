<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Technical background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1 – 1.5 pages (600 – 900 words)
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Standard Softmax Attention

For query-key scores $s = Q K^\top / \sqrt{d_h}$, softmax produces:

$$
p_i = \frac{e^{s_i}}{\sum_j e^{s_j}}, \quad p_i > 0, \quad \sum_i p_i = 1
\tag{3.1}
$$

Two properties matter here:

1. **Strict positivity.** Every $p_i > 0$; no exact zeros.
2. **Max-entropy solution.** Softmax solves:

$$
\text{softmax}(s) = \arg\max_{p \in \Delta^{d-1}} \langle p, s \rangle + H(p)
\tag{3.2}
$$

where $H$ is Shannon entropy. The entropy regularization spreads mass 
uniformly.

## 3.2 The α-Entmax Family

Replacing Shannon entropy with **Tsallis entropy** $H_\alpha^T$:

$$
H_\alpha^T(p) = \begin{cases}
\frac{1}{\alpha (\alpha - 1)} \sum_i (p_i - p_i^\alpha) & \alpha \neq 1 \\
-\sum_i p_i \log p_i & \alpha = 1
\end{cases}
\tag{3.3}
$$

gives the α-entmax family:

$$
\text{α-entmax}(s) = \arg\max_{p \in \Delta^{d-1}} \langle p, s \rangle + H_\alpha^T(p)
\tag{3.4}
$$

**Closed form.** For $\alpha > 1$:

$$
\text{α-entmax}(s)_i = \max\!\left(0, (\alpha - 1) s_i - \tau\right)^{1/(\alpha - 1)}
\tag{3.5}
$$

where $\tau$ is chosen so that $\sum_i p_i = 1$.

**Support.** The value $\tau$ determines the **support** of the output:

$$
S = \{i : (\alpha - 1) s_i > \tau\}
\tag{3.6}
$$

For $\alpha = 1.5$: $p_i \propto \max(0, 0.5 s_i - \tau)^2$.

## 3.3 Bisection for τ

Solving for $\tau$ is a one-dimensional root-finding problem:

$$
f(\tau) = \sum_i \max(0, (\alpha - 1) s_i - \tau)^{1/(\alpha - 1)} - 1 = 0
\tag{3.7}
$$

$f$ is monotone decreasing in $\tau$: at $\tau = -\infty$, $f = +\infty$; 
at $\tau = +\infty$, $f = -1$. Bisection converges in $\log_2(1/\epsilon)$ 
iterations. Our implementation uses 25 iterations, giving precision 
$\sim 10^{-7}$.

**Initial bounds.**

- $\tau_\text{hi} = 0$ (a natural upper bound since scores are typically 
  shifted so max = 0).
- $\tau_\text{lo} = \min_i (\alpha - 1) s_i - 1$ (a lower bound below which 
  all weights are positive).

## 3.4 Numerical Stability

**Score shifting.** To avoid overflow in $(\alpha - 1) s_i$, we shift scores 
by subtracting $\max_i s_i$. This preserves the argmax but bounds the 
exponent.

**Fallback.** If the support is empty (all weights zero), the output would be 
ill-defined. We fall back to softmax on the same masked scores.

## 3.5 Attention Sinks

**Definition.** Xiao et al. (2024) define attention sinks as tokens that 
receive high attention regardless of query content. In pretrained transformers, 
the first few tokens act as sinks because:

1. Positional embeddings for position 0 are unique.
2. Token 0 is often `[CLS]` or `<s>`, a natural aggregator.

**Empirical observation.** Removing sink tokens causes measurable accuracy 
degradation. Preserving them (first 4 in our implementation) maintains 
stability.

## 3.6 Local Splash Radius

For causal decoders, we restrict attention to:

$$
\text{mask}(i, j) = \begin{cases}
1 & \text{if } 0 \leq i - j \leq W \\
1 & \text{if } j < S \\
0 & \text{otherwise}
\end{cases}
\tag{3.8}
$$

where $W$ is the splash radius (default 32) and $S = 4$ is the sink count. 
Non-causal (bidirectional) models use the full attention (no windowing).

## 3.7 LoRA Adapters

LoRA (Hu et al., 2021) parameterizes a weight update:

$$
W_\text{effective} = W_\text{frozen} + B A
\tag{3.9}
$$

with $A \in \mathbb{R}^{r_L \times d_\text{in}}$, $B \in \mathbb{R}^{d_\text{out} \times r_L}$, 
$B = 0$ at initialization.

**In AdaSplash.** LoRA is added to each attention projection. The 
initial forward pass equals the pretrained model (since $B = 0$).

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $s$ | Raw attention scores |
| $p$ | Attention probability distribution |
| $\alpha$ | Entmax parameter |
| $\tau$ | Lagrange multiplier |
| $W$ | Splash radius |
| $S$ | Sink token count |
| $r_L$ | LoRA rank |
| $A, B$ | LoRA factors |
| $H, H_{kv}$ | Query, KV head counts |
| $d_h$ | Per-head dimension |
