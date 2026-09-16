# 2. Related Work

## 2.1 Mixture-of-Experts

**Original MoE.** Jacobs et al. (1991) introduced MoE as a mixture of 
"expert" networks with a "gating" network. Jordan & Jacobs (1994) extended 
to hierarchical MoE.

**Sparsely-Gated MoE.** Shazeer et al. (2017) scaled MoE to 137B parameters 
using noisy top-$k$ gating with load balancing. REAP-MoE's routing follows 
this design.

**Switch Transformer.** Fedus et al. (2021) simplify to top-1 routing with 
simplified load-balancing loss. This is the closest to our approach.

**GShard.** Lepikhin et al. (2021) use top-2 routing with auxiliary loss 
at scale. REAP-MoE uses top-$k$ routing with a similar auxiliary loss.

**Expert Choice.** Zhou et al. (2022) invert the routing: experts choose 
tokens. This naturally balances load but has different properties.

## 2.2 Load Balancing

**Auxiliary loss.** Shazeer et al. (2017) proposed 
$\mathcal{L}_\text{aux} = E \sum_i \rho_i \pi_i$. This is exactly what REAP-MoE 
uses.

**Z-loss.** Zoph et al. (2022) add a regularizer $\mathcal{L}_z = \log^2 \sum_i e^{\text{logit}_i}$ 
to stabilize router outputs.

**Router jitter.** Adding noise to router logits during training (as we do).

**Batch prioritization.** ByCheng et al. (2022) rank tokens by max routing 
probability. Not used in REAP-MoE.

## 2.3 Low-Rank Factorization for MoE

**Expert factorization.** Recent work has explored factorizing experts for 
parameter-efficient MoE. Chen et al. (2023) use SVD on expert weights. 
REAP-MoE uses the same idea but with a different rank schedule.

**MoE-LoRA.** Zadouri et al. (2023) add LoRA adapters to experts. REAP-MoE 
uses full SVD factorization of experts, not LoRA.

## 2.4 Position of Our Work

| Method | Routing | Load-balance | Expert compression |
|---|---|---|---|
| Shazeer et al. (2017) | Noisy top-k | Yes | None |
| Switch Transformer | Top-1 | Yes | None |
| GShard | Top-2 | Yes | None |
| Expert Choice | Expert-pick | Implicit | None |
| **REAP-MoE** | **Noisy top-k** | **Yes** | **SVD** |

**Key distinction.** REAP-MoE combines standard noisy top-$k$ routing with 
SVD-factorized experts, uniquely addressing both parameter bloat and load 
imbalance.

---