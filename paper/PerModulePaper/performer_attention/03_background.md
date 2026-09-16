<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 03_background.md
     PURPOSE: Background
     DEPENDENCIES: 02_related.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 3. Background

## 3.1 Softmax Attention

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right)V
\tag{3.1}
$$

**Cost:** $\mathcal{O}(L^2 d)$.

## 3.2 Kernel Trick

Reformulate softmax as a kernel:

$$
\text{softmax}(q_i^\top k_j / \sqrt{d}) = \frac{\exp(q_i^\top k_j / \sqrt{d})}{\sum_j \exp(q_i^\top k_j / \sqrt{d})}
\tag{3.2}
$$

Approximate $\exp(q^\top k) \approx \phi(q)^\top \phi(k)$:

$$
\text{Attn}_i \approx \frac{\phi(q_i)^\top \sum_j \phi(k_j) v_j^\top}{\phi(q_i)^\top \sum_j \phi(k_j)}
\tag{3.3}
$$

## 3.3 Random Fourier Features (RFF)

For a shift-invariant kernel $k(x, y) = k(x - y)$:

$$
k(x - y) = \int p(\omega) e^{i \omega^\top (x - y)} d\omega
\tag{3.4}
$$

Sample $\omega_1, \dots, \omega_F$ from $p(\omega)$:

$$
k(x - y) \approx \frac{1}{F} \sum_{i=1}^{F} e^{i \omega_i^\top (x - y)}
\tag{3.5}
$$

**Real form:**

$$
\phi(x) = \frac{1}{\sqrt{F}} [\cos(\omega_1^\top x), \dots, \cos(\omega_F^\top x), \sin(\omega_1^\top x), \dots, \sin(\omega_F^\top x)]
\tag{3.6}
$$

## 3.4 FAVOR+ Feature Map

Choromanski et al. (2021) propose the **positive** version:

$$
\phi(x) = \frac{1}{\sqrt{F}} \exp\!\left(\omega^\top x - \frac{\|x\|^2}{2}\right)
\tag{3.7}
$$

where $\omega$ is **orthogonal** (rows are orthonormal), which reduces the 
variance of the estimator.

**Why positive?** Softmax weights are always positive. Standard RFF can 
produce negative estimates, which break the positivity invariant.

## 3.5 Numerical Stability

For large $\|x\|$, $\exp(\omega^\top x - \|x\|^2/2)$ can overflow. The 
standard trick:

$$
\phi(x) = \exp\!\left(\omega^\top x - \max(\omega^\top x) - \frac{\|x\|^2}{2}\right)
\tag{3.8}
$$

The max-subtraction removes the largest value, keeping the exponent 
bounded.

## 3.6 GQA and RoPE

**GQA:** $H_{kv} < H$. Handled via `repeat_kv`.

**RoPE:** Rotary position embeddings. Orthogonal to the kernel 
approximation.

## 3.7 LoRA

Standard (Hu et al., 2021).

## 3.8 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Head dimension |
| $H$ | Number of heads |
| $F$ | Number of features |
| $\omega$ | RFF projection |
| $\phi$ | Feature map |
| $r_L$ | LoRA rank |

---