# 1. Introduction

## 1.1 Motivation

Standard softmax attention (Vaswani et al., 2017) computes a dense matrix 
of $L \times L$ scores, making both time and memory $\mathcal{O}(L^2)$. 
**Linear attention** (Katharopoulos et al., 2020; Choromanski et al., 2021) 
replaces the softmax with a kernel feature map $\phi$:

$$
\text{Attn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) \phi(K)^\top \mathbf{1}}
\tag{1.1}
$$

This is $\mathcal{O}(L)$ in both time and memory because the **associative** 
product $\phi(K)^\top V$ can be precomputed once.

**The problem.** Linear attention often fails to match softmax accuracy. 
Causes include:

1. **Unstable kernels.** Simple feature maps (e.g., $\text{elu}(x) + 1$) do 
   not approximate softmax well.
2. **No positional control.** Without re-weighting, all keys contribute 
   equally to the aggregated context.
3. **Distribution shift.** Pretrained attention weights expect softmax 
   geometry; linear attention geometry differs.

CHELA-SLAY addresses all three via three components:

1. **Spherical normalization** stabilizes the kernel input.
2. **Yat-Kernel** gives a richer feature space.
3. **Adaptive gating** blends linear with dense so the model can fall back 
   to softmax when needed.

## 1.2 Problem Statement

Given a pretrained transformer with softmax attention $\mathcal{A}$, we seek 
a replacement $\mathcal{A}'$ such that:

1. **Linear complexity for the linear path.** The linear stream runs in 
   $\mathcal{O}(L)$ time and memory.
2. **Bounded approximation error.** A dense stream provides an anchor; the 
   gate can weight it up if needed.
3. **Zero-shot preservation.** At initialization, $\mathcal{A}'$ produces 
   the same output as $\mathcal{A}$.
4. **Multi-architecture.** DistilBERT, RoBERTa, GPT-2, Qwen3.5.
5. **Causal-safe for decoders.**

## 1.3 Our Approach

CHELA-SLAY computes two parallel streams:

### 1.3.1 Dense Stream

Standard softmax attention:

$$
O^\text{dense} = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.2}
$$

This is our anchor — it preserves pretrained behavior.

### 1.3.2 Linear Stream

We spherical-normalize Q, K, apply a Yat-Kernel feature map, and compute 
linear attention:

$$
\hat{Q} = \frac{Q}{\|Q\|_2 + \epsilon}, \quad
\hat{K} = \frac{K}{\|K\|_2 + \epsilon}
\tag{1.3}
$$

$$
\phi(x) = \frac{1}{F^{1/4}} [\cos(x \Omega) \,\|\, \sin(x \Omega)]
\tag{1.4}
$$

$$
O^\text{linear} = \frac{\phi(\hat{Q}) \left[ \phi(\hat{K})^\top V \right]}{\phi(\hat{Q}) \left[ \phi(\hat{K})^\top \mathbf{1} \right] + \epsilon}
\tag{1.5}
$$

### 1.3.3 Adaptive Gate

A learnable per-head, per-token gate blends the two streams:

$$
g = \sigma(W_g X + b_g), \quad
\text{Output} = W_O \left( g \odot O^\text{dense} + (1 - g) \odot O^\text{linear} \right)
\tag{1.6}
$$

We initialize $W_g = 0$ and $b_g = 5.0$, so 
$g \approx 0.993$ at step 0 — favoring the dense path.

## 1.4 Contributions

- **A hybrid dense + linear attention block.** CHELA-SLAY combines a 
  softmax anchor with a stabilized linear stream.
- **Spherical normalization + Yat-Kernel.** A stabilized feature map for 
  linear attention.
- **Adaptive per-head gating.** The model learns how much to trust each 
  stream, initialized toward dense.
- **Multi-architecture support.** Weight transfer from DistilBERT, RoBERTa, 
  GPT-2, and Qwen3.5.
- **Honest reporting of limitations.** We explicitly document the cumsum 
  memory footprint and the short-sequence slowdown.
- **Full reproducibility.** Code, weights, and scripts released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on linear 
attention, RFF, and spherical normalization. Section 4 presents CHELA-SLAY. 
Section 5 describes integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes accuracy and memory. Section 9 
discusses practical guidance. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $d_h$ | Head dimension |
| $H$ | Number of heads |
| $H_{kv}$ | KV head count |
| $F$ | Number of RFF features |
| $\Omega$ | RFF projection matrix |
| $\phi(\cdot)$ | Yat-Kernel feature map |
| $g$ | Adaptive gate |
| $b_g$ | Gate bias |
| $\hat{Q}, \hat{K}$ | Spherically normalized Q, K |
| $\epsilon$ | Numerical stability constant |

---