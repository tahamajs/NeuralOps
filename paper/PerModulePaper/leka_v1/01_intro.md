# 1. Introduction

## 1.1 Motivation

Standard transformer attention (Vaswani et al., 2017) computes:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

This scales as $\mathcal{O}(L^2)$ in both time and memory, making 
long-context inference expensive.

**Linear attention** replaces softmax with a kernel feature map:

$$
\text{Attn}(Q, K, V) \approx \frac{\phi(Q) \left(\phi(K)^\top V\right)}{\phi(Q) \left(\phi(K)^\top \mathbf{1}\right)}
\tag{1.2}
$$

Since $\phi(K)^\top V \in \mathbb{R}^{F \times d_h}$ is computed once and 
reused, the total complexity is $\mathcal{O}(L F d_h)$ — **linear in L**.

The quality of linear attention depends entirely on how well $\phi$ 
approximates the softmax kernel:

$$
\kappa(q, k) = \exp(q^\top k)
\tag{1.3}
$$

## 1.2 The Kernel Approximation Problem

Three families of feature maps have been studied:

1. **Performer (Choromanski et al., 2021):** Positive Orthogonal Random 
   Features (PORF), which approximate $\exp(q^\top k)$ with positive 
   features.
2. **Linear Transformer (Katharopoulos et al., 2020):** $\phi(x) = \text{elu}(x) + 1$, 
   which corresponds to a specific kernel that is **not** the softmax 
   kernel.
3. **cosFormer (Qin et al., 2022):** uses $\text{ReLU}$ with a cosine 
   reweighting.

We take a fourth approach: **Random Fourier Features (RFF)** for the 
**Gaussian RBF kernel** $\exp(-\|q - k\|^2 / 2)$:

$$
\kappa(q, k) \approx \phi(q)^\top \phi(k), \quad
\phi(x) = \frac{1}{\sqrt{F}} \left[ \cos(x W), \sin(x W) \right]
\tag{1.4}
$$

with $W_{ij} \sim \mathcal{N}(0, 1)$. This is the **real part** of the 
classical RFF estimator (Rahimi & Recht, 2007).

## 1.3 Our Approach: LEKAv1

LEKAv1 has four components:

### 1.3.1 Head-wise LayerNorm

Before projection, we normalize each head:

$$
\tilde{x} = \text{LayerNorm}(x) \in \mathbb{R}^{d_h}
\tag{1.5}
$$

This centers and scales the input, making the random projection more 
stable.

### 1.3.2 QR-Orthogonalized Projection

The projection matrix $W \in \mathbb{R}^{d_h \times F}$ is initialized from 
the QR decomposition of a random Gaussian:

$$
W = Q, \quad \text{QR}(G) = QR, \quad G \sim \mathcal{N}(0, I)
\tag{1.6}
$$

This ensures orthonormal columns, reducing variance in the kernel 
approximation.

### 1.3.3 cos/sin Feature Map

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\tilde{x} W) \| \sin(\tilde{x} W)]
\tag{1.7}
$$

which gives $\phi(x) \in \mathbb{R}^{2F}$.

### 1.3.4 Preservation Gate

A single learnable scalar $\alpha = \sigma(w)$ blends attention output 
with the input:

$$
\text{out} = \alpha \odot A + (1 - \alpha) \odot x
\tag{1.8}
$$

This provides a **residual safety net**: if the kernel approximation is 
poor, the model can lean toward the identity.

## 1.4 Contributions

- **A production-hardened RFF attention.** We implement cos/sin feature 
  maps with QR-orthogonalized projections, causal cumsum, and 
  bidirectional global KV.
- **Head-wise LayerNorm before projection.** We normalize the head input 
  before random projection, which stabilizes the kernel approximation.
- **A preservation gate.** A scalar gate controls mixing with the residual.
- **Multi-architecture support.** DistilBERT, RoBERTa, GPT-2, Qwen3.5 
  with GQA and RoPE.
- **An honest evaluation.** We report both the accuracy and the parameter 
  cost.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on kernels, 
RFF, and linear attention. Section 4 presents LEKAv1. Section 5 describes 
integration. Section 6 details the setup. Section 7 reports results. 
Section 8 analyzes the kernel approximation. Section 9 discusses guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden size |
| $H$ | Heads |
| $d_h$ | Head dim |
| $F$ | Feature dimension |
| $\phi$ | Kernel feature map |
| $W$ | Random projection |
| $Q, K, V$ | Attention projections |
| $\alpha$ | Preservation gate |

---