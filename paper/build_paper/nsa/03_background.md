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
S = \frac{QK^\top}{\sqrt{d_h}}, \quad
P = \text{softmax}(S), \quad
O = PV
\tag{3.1}
$$

**Complexity.** $\mathcal{O}(L^2)$ memory and time.

## 3.2 Fixed Sparse Attention

Restrict attention to a fixed subset $\mathcal{S}_{ij}$:

$$
P_{ij} = \frac{\mathbb{1}[\mathcal{S}_{ij}] \exp(s_{ij})}{\sum_{j'} \mathbb{1}[\mathcal{S}_{ij'}] \exp(s_{ij'})}
\tag{3.2}
$$

**Sliding window.** $\mathcal{S}_{ij} = \mathbb{1}[|i - j| \leq w]$.

**Causal.** $\mathcal{S}_{ij} = \mathbb{1}[j \leq i]$ (and optionally band-limited).

**Complexity.** $\mathcal{O}(L w)$ memory and time.

## 3.3 Top-K Selection

Given scores $s_i \in \mathbb{R}^L$, the top-$k$ set is:

$$
\text{TopK}(s_i) = \{j : s_{ij} \geq \tau_i\}
\tag{3.3}
$$

where $\tau_i$ is the $k$-th largest score.

**Non-differentiability.** The top-$k$ indicator has zero gradient almost 
everywhere. However, in **inference-only** settings (which is our focus), 
this does not matter.

## 3.4 Union of Local and Global Masks

NSA uses:

$$
\mathcal{S}_{ij} = \mathbb{1}[|i - j| \leq w] \lor \mathbb{1}[j \in \text{TopK}_i]
\tag{3.4}
$$

The mask is applied **before** softmax:

$$
P_{ij} = \frac{\mathcal{S}_{ij} \exp(s_{ij})}{\sum_{j'} \mathcal{S}_{ij'} \exp(s_{ij'})}
\tag{3.5}
$$

## 3.5 Causal Masking

For decoders:

$$
\mathcal{S}_{ij}^\text{causal} = \mathcal{S}_{ij} \land \mathbb{1}[j \leq i]
\tag{3.6}
$$

Applied **after** top-$k$ so that top-$k$ never selects future positions.

## 3.6 Padding

Padding keys must be excluded:

$$
\mathcal{S}_{ij}^\text{final} = \mathcal{S}_{ij}^\text{causal} \land \mathbb{1}[\text{valid}_j]
\tag{3.7}
$$

## 3.7 Attention Density

The density of attention is:

$$
\rho = \frac{\mathbb{E}_i[|\mathcal{S}_i|]}{L} \approx \frac{2w + 1 + k}{L}
\tag{3.8}
$$

For $w = 32$, $k = 16$, $L = 128$: $\rho \approx \frac{81}{128} = 0.63$.

For $w = 32$, $k = 16$, $L = 2048$: $\rho \approx 0.04$.

## 3.8 LoRA Recovery

LoRA (Hu et al., 2021) adds trainable adapters:

$$
W_\text{eff} = W_\text{frozen} + \frac{\alpha}{r} B A
\tag{3.9}
$$

NSA uses LoRA to adapt the frozen projections to the sparse structure.

## 3.9 Notation Summary

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $w$ | Window size |
| $k$ | Top-$k$ count |
| $\mathcal{S}_i$ | Selected keys at query $i$ |
| $\mathcal{W}_i$ | Local window |
| $\rho$ | Attention density |
| $\tau_i$ | Top-$k$ threshold |

---