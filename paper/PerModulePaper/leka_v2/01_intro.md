# 1. Introduction

## 1.1 Motivation

Long-context inference is dominated by attention's $\mathcal{O}(L^2)$ cost. 
**Linear attention** (Katharopoulos et al., 2020) reduces this to 
$\mathcal{O}(L)$ via a kernel feature map $\phi$:

$$
\text{Attn}(Q, K, V) \approx \frac{\phi(Q)(\phi(K)^\top V)}{\phi(Q)(\phi(K)^\top \mathbf{1})}
\tag{1.1}
$$

However, linear attention has two limitations:

1. **Uniform aggregation.** The KV state $\phi(K)^\top V$ is a single global 
   summary. Queries cannot selectively attend to specific tokens — they 
   only see the aggregate.
2. **Kernel approximation error.** RFF with $F$ features gives $\mathcal{O}(1/F)$ 
   approximation error; small $F$ degrades accuracy.

**LEKAv1** (companion work) addressed the second via QR-orthogonalized 
projections. **LEKAv2** additionally addresses the first.

## 1.2 Problem Statement

Given $X \in \mathbb{R}^{B \times L \times d}$, we seek a hybrid attention 
that:

1. **Scales linearly** in $L$.
2. **Retains selective focus** on important tokens.
3. **Approximates a smooth kernel** (Gaussian RBF).
4. **Handles masking** without NaN or uniform collapse.
5. **Transfers pretrained weights** zero-shot.

## 1.3 Our Approach: LEKAv2

LEKAv2 combines four components:

### 1.3.1 Dual-Scale Random Features

Two independent RFF projections with dimensions $F$ and $F/2$:

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(x W_1) \| \sin(x W_1)] 
\| \frac{1}{\sqrt{F/2}} [\cos(x W_2) \| \sin(x W_2)]
\tag{1.2}
$$

with $W_1 \in \mathbb{R}^{d_h \times F}$, $W_2 \in \mathbb{R}^{d_h \times F/2}$.

### 1.3.2 Landmark Distillation

A learned scorer $s_\theta$ selects $L_\text{max} = \min(\rho L, L)$ 
landmark tokens:

$$
\mathcal{L}_\text{idx} = \text{TopK}(s_\theta(\phi(K)), L_\text{max})
\tag{1.3}
$$

Position 0 (`[CLS]`) is always retained.

### 1.3.3 Softpick

A ReLU-damped softmax that handles masked rows gracefully:

$$
\text{softpick}(x)_i = \frac{\text{ReLU}(e^{(x_i - m)/T} - e^{-m/T})}{\sum_j \text{ReLU}(e^{(x_j - m)/T} - e^{-m/T})}
\tag{1.4}
$$

with fallback to uniform when all entries are zero.

### 1.3.4 Local-Global Fusion

A per-head learnable gate blends local sliding-window attention and 
global landmark attention:

$$
\text{out} = (1 - g) \cdot \text{Local} + g \cdot \text{Global}, \quad g = \sigma(w_h)
\tag{1.5}
$$

with $w_h = -2$ initially, biasing toward local attention.

## 1.4 Contributions

- **Dual-scale RFF.** We extend RFF with a second projection at $F/2$, 
  enriching the kernel representation.
- **Learned landmark distillation.** A small scorer selects landmarks 
  whose features are used in the global branch.
- **Softpick.** A masked-attention-stable variant of softmax.
- **Local-global fusion.** Per-head gate blends local and global paths.
- **Per-head correction matrix.** $W_\text{corr} \in \mathbb{R}^{H \times d_h \times d_h}$ 
  applied to landmark values.
- **Multi-architecture support.** DistilBERT, RoBERTa, GPT-2, Qwen3.5 
  with GQA and RoPE.
- **An honest evaluation.** We report both accuracy and parameter cost.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on kernels, 
RFF, softmax variants, and landmark methods. Section 4 presents LEKAv2. 
Section 5 describes integration. Section 6 details the setup. Section 7 
reports results. Section 8 analyzes components. Section 9 discusses 
guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden size |
| $H$ | Heads |
| $d_h$ | Head dim |
| $F$ | Primary feature dim |
| $\rho$ | Landmark ratio |
| $L_\text{max}$ | Landmark count |
| $\phi$ | Feature map |
| $s_\theta$ | Landmark scorer |
| $W_\text{corr}$ | Correction matrix |
| $g$ | Local gate |
| $T$ | Softpick temperature |

---