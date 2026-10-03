# 1. Introduction

## 1.1 Motivation

Mixture of Experts (MoE) has emerged as a powerful technique for scaling 
transformer capacity without proportional compute (Shazeer et al., 2017; 
Fedus et al., 2022; Lepikhin et al., 2021). In a standard MoE, a router 
selects top-$k$ experts per token, and only the selected experts run forward 
passes. This gives the model more "knowledge slots" while keeping FLOPs 
roughly constant.

**The pretrained-model problem.** Most MoE methods are applied during 
pretraining: experts and routers are initialized randomly and trained 
end-to-end on a large corpus. Applying MoE **post hoc** to an already 
pretrained transformer is more delicate. Random initialization of the MoE 
branch perturbs the pretrained representation, causing catastrophic 
forgetting and necessitating extensive recovery training.

**A three-part solution.** BAM-UMoE solves this with:

1. **Zero-initialized experts.** Every expert's final output projection 
   starts at exactly zero. The MoE branch therefore contributes nothing 
   to the forward pass at step 0.

2. **Negative gating bias.** A learnable gate with initial bias $-3.0$ 
   produces $\sigma(-3) \approx 0.047$, meaning the MoE branch is attenuated 
   by ~95% at start. As training progresses, the gate learns to open.

3. **Frozen base attention.** The pretrained $W_Q, W_K, W_V, W_O$ are copied 
   exactly and frozen. LoRA adapters provide trainable capacity, and the MoE 
   branch provides additional routing capacity.

Together, these ensure that **step 0 is mathematically identical to the 
pretrained model**.

## 1.2 Problem Statement

Given a pretrained attention module with projections 
$W_Q, W_K, W_V, W_O \in \mathbb{R}^{d \times d}$ and a target task, we seek 
an augmented attention module $\mathcal{A}'$ that:

1. **Preserves step-0 behavior.** $\mathcal{A}'(X) = \mathcal{A}(X)$ for 
   all inputs $X$ at initialization.
2. **Adds capacity.** The augmented module has more trainable parameters 
   than the original, but not more activation FLOPs.
3. **Supports recovery fine-tuning.** Only a small fraction of parameters 
   is trainable, minimizing catastrophic forgetting.
4. **Works across architectures.** Flat (DistilBERT), nested (RoBERTa), 
   causal (GPT-2), and GQA+RoPE (Qwen/LLaMA).

**Note.** This is **not** a compression method. Parameter count increases 
substantially; the goal is capacity expansion with stable recovery.

## 1.3 Our Approach

BAM-UMoE augments the **value pathway** of attention with a top-$k$ MoE 
branch:

$$
V' = V_{\text{base}} + \alpha \cdot \text{MoE}(X)
\tag{1.1}
$$

where $X$ is the input hidden states, $V_\text{base} = X W_V^\top$ is the 
frozen base value projection, and MoE$(X)$ is a top-$k$ expert mixture 
producing a refinement signal of the same shape. The coefficient $\alpha = 0.1$ 
scales the MoE contribution conservatively.

**Key property.** At initialization, MoE$(X) = 0$ because every expert's 
output projection is zero. Step 0 is exact.

## 1.4 Contributions

- **A three-part step-0 preservation mechanism.** Zero-init + negative 
  gating bias + frozen base. Section 4.3.
- **A top-2 MoE attention refinement.** A vectorized expert mixture applied 
  to the value pathway with an auxiliary load-balancing loss. Section 4.2.
- **Broad architecture support.** Multi-architecture weight extraction plus 
  nested encoder detection, GQA, RoPE with partial dims, and cross-attention. 
  Section 5.
- **Honest disclosure.** We clearly state that BAM-UMoE **increases** 
  parameter count. Section 7.2.
- **Full reproducibility.** Code, checkpoints, and evaluation scripts released 
  under MIT.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on MoE, LoRA, 
and step-0 preservation. Section 4 presents the method. Section 5 describes 
architecture integration. Section 6 details the experimental setup. Section 7 
reports results. Section 8 analyzes the step-0 property and ablation 
contributions. Section 9 discusses usage. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $W_Q, W_K, W_V, W_O$ | Attention projections |
| $E$ | Number of experts |
| $k$ | Top-k selection |
| $r_L$ | LoRA rank |
| $\alpha$ | MoE contribution scale |
| $g$ | Learnable gate scalar |
| $b_g$ | Gate bias (initialized $-3.0$) |
| $\mathcal{L}_\text{aux}$ | Auxiliary load-balancing loss |
| $H, H_{kv}$ | Query, KV head counts |
| $d_h$ | Per-head dimension |

---