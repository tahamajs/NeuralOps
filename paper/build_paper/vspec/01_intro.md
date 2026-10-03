# 1. Introduction

## 1.1 Motivation

Attention in transformers is parameter-expensive: the four projections 
$W_Q, W_K, W_V, W_O$ contribute $4d^2$ parameters per layer. For a 
768-dimensional model, that is 2.36M per layer, or ~28M across 12 layers.

Three orthogonal avenues for improvement have emerged:

1. **Low-rank factorization** — replace $W$ with $UV$ (A3, SOA Fused, 
   Adaptive SVD).
2. **Token sparsity** — attend to only the important tokens (H2O, 
   StreamingLLM, TCA).
3. **Head-level modulation** — weight or prune heads (Squeezing Heads, 
   Selective Attention).

Each has been studied independently. We ask: **can they be combined, and 
can the gating be trained via variational inference?**

## 1.2 Problem Statement

Given a pretrained attention block with projections $\{W_Q, W_K, W_V, W_O\}$ 
and input sequence $X \in \mathbb{R}^{B \times L \times d}$, we seek a 
replacement block $\mathcal{B}(X)$ that:

1. **Compresses** each projection via truncated SVD.
2. **Routes** tokens to a subset of size $k < L$.
3. **Gates** singular values with a stochastic, input-dependent weight.
4. **Gates** heads with a per-example importance.
5. **Transfers** pretrained weights zero-shot.
6. **Trains** only adapters, gates, and router.

## 1.3 Our Approach: VSpec++

VSpec++ has four components:

### 1.3.1 Token Router

Given $X$ and a validity mask, a learned saliency network computes:

$$
s_i = \sigma(f_\theta(x_i)) \in [0, 1]
\tag{1.1}
$$

We force $s_0 = 10^4$ (retaining `[CLS]`), mask padding to $-\infty$, and 
select the top-$k$ indices. Then gather:

$$
X_\text{routed} = \text{gather}(X, \mathcal{S}), \quad k = \lceil \rho L \rceil
\tag{1.2}
$$

### 1.3.2 Variational Spectral Gate

For each projection, we compute a **posterior over singular value 
importances**:

$$
q(g | x) = \mathcal{N}\!\left(\mu(x), \text{diag}(\sigma^2(x))\right)
\tag{1.3}
$$

where $\mu(x) = f_\mu(\text{pool}(x))$ and $\log \sigma(x) = f_\sigma(\text{pool}(x))$. 
We sample:

$$
g = \sigma(\mu + \sigma \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)
\tag{1.4}
$$

At inference, $g = \sigma(\mu)$.

### 1.3.3 Head Gate

A small network computes per-head importances:

$$
h = \sigma(f_h(\text{pool}(x))) \in [0, 1]^H
\tag{1.5}
$$

These multiply the attention logits.

### 1.3.4 Factorized Projections

Each projection is:

$$
W_\text{eff} = U \cdot \text{diag}(S \odot g) \cdot V^\top + B A
\tag{1.6}
$$

where $U, S, V$ are SVD factors (frozen), $g$ is the spectral gate, and 
$B A$ is a LoRA adapter.

## 1.4 Contributions

- **A variational spectral gate.** We introduce a per-singular-value 
  posterior trained via reparameterization.
- **Three-level token-adaptive gating.** Token routing, spectral gating, 
  and head gating are jointly trained.
- **Zero-shot SVD initialization.** Factor initialization requires no 
  calibration.
- **A production-hardened implementation.** Handles DistilBERT, RoBERTa, 
  GPT-2, and Qwen3.5, with masks, GQA, RoPE, and nested encoders.
- **Full reproducibility.** Code, weights, and scripts released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on SVD, 
variational inference, and LoRA. Section 4 presents VSpec++. Section 5 
describes integration. Section 6 details the setup. Section 7 reports 
results. Section 8 analyzes the gates. Section 9 discusses guidance. 
Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $k$ | Routed sequence length |
| $\rho$ | Keep ratio |
| $d$ | Hidden size |
| $H$ | Heads |
| $r$ | SVD rank |
| $r_L$ | LoRA rank |
| $\mu, \sigma$ | Variational parameters |
| $g$ | Spectral gate |
| $h$ | Head gate |
| $s$ | Token saliency |

---