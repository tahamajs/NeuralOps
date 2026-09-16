# 2. Related Work

## 2.1 Low-Rank Attention

**SOA Fused Attention** and **Adaptive SVD Attention** factorize Q/K/V 
(and possibly O) via truncated SVD. **A3** and **AASVD** use 
activation-weighted SVD requiring calibration. VSpec++ differs by 
combining **spectral modulation** with **token routing**.

## 2.2 Token Sparse Attention

**H2O** (Zhang et al., 2023) evicts KV tokens by accumulated attention. 
**StreamingLLM** (Xiao et al., 2024) preserves attention sinks. **TCA** 
(companion work) uses a learnable saliency net with `[CLS]` preservation. 
VSpec++ uses a similar router, combined with the other gates.

## 2.3 Head Sparsity

**Squeezing Heads** (companion) distills heads. **Selective Attention** 
(Lee et al., 2019) prunes heads. **HeadGate** in VSpec++ provides per-example 
soft weighting rather than hard pruning.

## 2.4 Variational Methods in Attention

**Variational Dropout** (Kingma et al., 2015; Molchanov et al., 2017) 
learns per-weight dropout rates via reparameterization. **VIB** (Alemi 
et al., 2017) uses variational bounds for representation learning. 
**Variational Attention** (Bahuleyan et al., 2018) uses a latent variable 
for attention. Our variational spectral gate is closest to Variational 
Dropout but applied to singular value importances.

## 2.5 Reparameterization Trick

**Kingma & Welling (2013)** introduced the reparameterization trick for 
training VAEs. **Rezende et al. (2014)** developed similar ideas 
independently. We use the standard Gaussian reparameterization:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\tag{2.1}
$$

## 2.6 Position of Our Work

| Method | SVD | Token route | Spectral gate | Head gate | Variational |
|---|---|---|---|---|---|
| SOA Fused | Yes | No | No | No | No |
| TCA | No | Yes | No | No | No |
| Squeezing Heads | No | No | No | Yes | No |
| Variational Dropout | No | No | No | No | Yes |
| **VSpec++** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

**Key distinction.** VSpec++ is the first to combine all four techniques 
in a single attention block.

---

## Comparison Table

| Method | Params | Token | Spectral | Variational |
|---|---|---|---|---|
| MHSA | $4d^2$ | Full | — | — |
| SOA Fused | $2d^2 + 2rd$ | Full | — | — |
| TCA | $4d^2$ | Top-$k$ | — | — |
| **VSpec++** | $2rd + \text{gates}$ | **Top-$k$** | **Yes** | **Yes** |

---