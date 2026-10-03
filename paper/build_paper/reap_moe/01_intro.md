# 1. Introduction

## 1.1 Motivation

Mixture-of-Experts (MoE) architectures (Shazeer et al., 2017; Fedus et al., 
2021) increase the effective capacity of a transformer by replacing the 
FFN with a set of $E$ expert networks and a router that selects $k$ experts 
per token. This allows a model to have $\mathcal{O}(E)$ more parameters while 
keeping per-token FLOPs at $\mathcal{O}(k)$.

However, MoE has three well-known problems:

1. **Parameter bloat.** With $E$ experts, the FFN block grows $E$-fold. For 
   DistilBERT with FFN of 2.36M parameters and $E = 8$ experts, this is 18.9M 
   parameters per layer, or 113M across six layers — nearly twice the entire 
   model.

2. **Router collapse.** Without regularization, the router often converges 
   to assigning all tokens to a small subset of experts (Shazeer et al., 2017; 
   Chi et al., 2022). This defeats the purpose of MoE.

3. **Training instability.** Load imbalance causes high-variance gradients 
   and unstable training, especially in the early epochs.

REAP-MoE addresses all three:

1. **SVD-factorized experts.** Each expert's linear layers are decomposed 
   into $U \Sigma V^\top$ with rank $r \ll \min(d_\text{in}, d_\text{out})$, 
   reducing per-expert parameters by $d_\text{in} d_\text{out} / (r(d_\text{in} + d_\text{out}))$.

2. **Load-balancing auxiliary loss.** We add $\mathcal{L}_\text{aux} = E \sum_i \rho_i \pi_i$ 
   where $\rho_i$ is the fraction of tokens routed to expert $i$ and $\pi_i$ is 
   the mean router probability for expert $i$. This is minimized when both 
   are uniform.

3. **Noise-injected routing.** During training, Gaussian noise is added to 
   router logits before top-$k$ selection, providing exploration.

## 1.2 Problem Statement

Given a pretrained transformer with an FFN block per layer, we seek a MoE 
replacement such that:

$$
\text{Capacity}(\mathcal{M}') > \text{Capacity}(\mathcal{M}), \quad
\text{Params}(\mathcal{M}') < E \cdot \text{Params}(\text{FFN})
\tag{1.1}
$$

while maintaining or improving downstream accuracy.

## 1.3 Contributions

- **SVD-factorized MoE experts** with per-expert compression.
- **Load-balancing loss** adapted from Switch Transformer.
- **Noise-injected top-$k$ routing** for exploration.
- **Unified attention + MoE module** with pre-LayerNorm around each.
- **Multi-architecture support** with weight transfer from pretrained models.
- **Honest evaluation**: we report both where REAP-MoE helps and where 
  parameter growth outweighs benefits.

## 1.4 Paper Organization

Section 2 reviews MoE, SVD, and load balancing. Section 3 provides background 
on MoE, top-$k$ routing, and SVD. Section 4 presents REAP-MoE. Section 5 
describes integration. Section 6 details setup. Section 7 reports results. 
Section 8 analyzes component contributions. Section 9 discusses practical 
use. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $E$ | Number of experts |
| $k$ | Top-$k$ selections |
| $d$ | Hidden dimension |
| $r$ | SVD rank |
| $\rho_i$ | Router utilization of expert $i$ |
| $\pi_i$ | Mean router probability for expert $i$ |
| $\mathcal{L}_\text{aux}$ | Auxiliary load-balancing loss |
| $\sigma$ | Noise standard deviation |
| $U, \Sigma, V$ | SVD factors |

---