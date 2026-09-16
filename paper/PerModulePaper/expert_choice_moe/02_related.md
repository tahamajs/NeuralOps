# 2. Related Work

## 2.1 Mixture of Experts

**Shazeer et al. (2017)** introduced sparsely-gated MoE with top-$k$ 
token-choice routing. **GShard** (Lepikhin et al., 2021) scaled to 600B 
parameters using expert parallelism. **Switch Transformer** (Fedus et al., 
2022) simplified to top-1 routing and introduced the auxiliary 
load-balancing loss:

$$
\mathcal{L}_\text{aux} = E \sum_{i=1}^{E} f_i P_i
\tag{2.1}
$$

**Expert Choice** (Zhou et al., 2022) reversed the routing direction:
each expert selects tokens. We adopt this formulation.

## 2.2 Expert Choice Routing

Expert-choice routing:

- **Guaranteed load balance.** Each expert processes exactly $k = T / E$ 
  tokens.
- **Dynamic per-token capacity.** Tokens can be selected by 0, 1, or many 
  experts.
- **No auxiliary loss.**

Zhou et al. (2022) show improved zero-shot transfer and stable training 
compared to token-choice. We bring this to attention augmentation.

## 2.3 MoE in Attention

**BAM-UMoE** (companion work) uses token-choice MoE on attention values. 
Our work uses expert-choice MoE on attention **outputs**. The routing 
mechanism and target differ.

## 2.4 Attention Augmentation

**LoRA** (Hu et al., 2021) adds low-rank deltas to frozen weights. 
**AdaLoRA** (Zhang et al., 2023) adaptively allocates rank. **Compacter** 
(Karimi Mahabadi et al., 2021) uses Kronecker adapters.

Our work is orthogonal to these — we add a full MoE block, not a low-rank 
adapter.

## 2.5 Zero-Init Stability

**Goyal et al. (2017)** show zero-init of last layers stabilizes training. 
**Fixup** (Zhang et al., 2019) uses scaled initialization. **ReZero** 
(Bachlechner et al., 2021) initializes residual gates to zero. We use both: 
zero-init expert output and negative gate bias.

## 2.6 Position of Our Work

| Method | Routing | Location | Guaranteed balance |
|---|---|---|---|
| Switch Transformer | Token-choice | FFN | ❌ (needs aux loss) |
| GShard | Token-choice | FFN | ❌ |
| Mixtral | Token-choice | FFN | ❌ |
| **Expert Choice** (Zhou) | **Expert-choice** | FFN | ✅ |
| BAM-UMoE | Token-choice | Attention V | ❌ |
| **ExpertChoiceMoE** | **Expert-choice** | **Attention output** | ✅ |

**Key distinction.** First method to apply **expert-choice MoE inside 
attention** with guaranteed load balancing.

---

## Comparison Table

| Method | Routing | Load balance | Zero-init | Cross-attn |
|---|---|---|---|---|
| Switch Transformer | Token | Aux loss | ❌ | N/A |
| Mixtral | Token | Aux loss | ❌ | N/A |
| BAM-UMoE | Token | Aux loss | ✅ | Skip |
| **ExpertChoiceMoE** | **Expert** | **Guaranteed** | ✅ | ✅ |

---