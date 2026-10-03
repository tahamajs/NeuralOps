# 1. Introduction

## 1.1 Motivation

Mixture-of-Experts (MoE) has become a standard building block for scaling 
transformers (Shazeer et al., 2017; Lepikhin et al., 2021; Fedus et al., 
2022). In the conventional **token-choice** formulation, a router produces 
per-token expert logits, and each token independently selects its top-$k$ 
experts. This has two practical issues:

1. **Load imbalance.** Some experts receive many tokens while others receive 
   few. The training-time fix is an auxiliary load-balancing loss, which 
   adds a hyperparameter and is often ineffective at high imbalance.
2. **Underutilization.** Tokens with ambiguous routing patterns may not 
   benefit from the experts best suited for them.

**Expert-choice routing** (Zhou et al., 2022) reverses the roles: each 
**expert** selects the $k$ tokens with the highest affinity. This 
provides:

- **Guaranteed perfect balance.** Every expert processes exactly 
  $k = \lceil T / E \rceil$ tokens, where $T$ is the total token count.
- **Dynamic per-token allocation.** Tokens that match many experts' 
  preferences receive more capacity; tokens that match few receive less.
- **No auxiliary loss needed.**

## 1.2 Problem Statement

Given a pretrained transformer with standard multi-head attention, we seek 
an augmented attention block $\mathcal{A}'$ such that:

1. **Step-0 stability.** At initialization, $\mathcal{A}'$ produces the 
   same output as $\mathcal{A}$.
2. **Bounded perturbation.** The MoE contribution is gated and bounded.
3. **Perfect load balancing.** Every expert processes the same number of 
   tokens.
4. **Dynamic capacity.** Tokens receive capacity proportional to their 
   importance.
5. **Cross-architecture.** DistilBERT, RoBERTa, GPT-2, Qwen3.5.
6. **Cross-attention safe.**

## 1.3 Our Approach

**ExpertChoiceMoEAttention** adds a **top-$k$ expert-choice MoE block** 
inside attention:

$$
\text{attn\_out} = \text{Attention}(Q, K, V)
\tag{1.1}
$$

$$
\text{MoE\_out} = \text{ExpertChoiceMoE}(\text{attn\_out})
\tag{1.2}
$$

$$
\text{final} = W_O \left( \text{attn\_out} + \gamma \cdot \text{MoE\_out} \right)
\tag{1.3}
$$

where $\gamma = \sigma(b_g)$ and $b_g = 0.1$ initially.

**Three key design choices:**

### 1.3.1 Zero-Init Experts

Every expert's final projection is zero-initialized:

$$
W_2^{(e)} = 0 \implies \text{MoE\_out} = 0 \text{ at step 0}
\tag{1.4}
$$

### 1.3.2 Negative Gate Bias

$$
\gamma = \sigma(0.1) \approx 0.8
\tag{1.5}
$$

### 1.3.3 Expert-Choice Routing

Each expert selects $k = \lceil T / E \rceil$ tokens:

$$
\text{Top-}k_e = \text{TopK}\!\left(\text{softmax}(\text{logits})[:, e], k\right)
\tag{1.6}
$$

Perfect load balancing by construction.

## 1.4 Contributions

- **Expert-choice MoE inside attention.** First method to combine expert-
  choice routing with attention augmentation.
- **Guaranteed load balancing.** No auxiliary loss needed.
- **Step-0 stability.** Zero-init + negative gate bias.
- **Multi-architecture.** DistilBERT, RoBERTa, GPT-2, Qwen.
- **Cross-attention support.**
- **Honest parameter reporting.** We state clearly that parameters increase.
- **Full reproducibility.** Code, weights, scripts.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on MoE, 
expert-choice routing, and attention. Section 4 presents 
ExpertChoiceMoEAttention. Section 5 describes integration. Section 6 details 
the setup. Section 7 reports results. Section 8 analyzes load balancing. 
Section 9 discusses trade-offs. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $E$ | Number of experts |
| $T = B \cdot L$ | Total tokens |
| $k = \lceil T / E \rceil$ | Tokens per expert |
| $b_g$ | MoE gate bias |
| $\gamma$ | MoE gate value |
| $H$ | Number of heads |
| $d_h$ | Head dim |

---