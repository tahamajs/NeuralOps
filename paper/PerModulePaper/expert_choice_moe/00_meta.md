# ExpertChoiceMoEAttention: Expert-Choice Routing Inside Multi-Head Attention

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/expert_choice_moe.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Mixture-of-Experts (MoE) layers have become a standard technique for scaling 
feed-forward networks without proportionally increasing compute. In 
conventional **token-choice** MoE, each token selects its top-$k$ experts 
and the resulting load is often highly imbalanced, requiring an auxiliary 
loss to stabilize training.

We adopt **expert-choice routing** (Zhou et al., 2022), in which the roles 
are reversed: each expert selects its top-$k$ **tokens**. This provides 
**guaranteed perfect load balancing** — every expert processes exactly 
$k = \lceil T / E \rceil$ tokens, where $T$ is the total token count and $E$ 
is the number of experts — and enables **dynamic per-token expert allocation** 
(tokens that match many experts' preferences receive more capacity).

We place the MoE block **inside multi-head attention** — on the attention 
output before the final output projection — with:

1. **Zero-init experts.** The final projection of every expert is initialized 
   to zero, so at step 0, the MoE contributes nothing and the attention 
   behaves exactly as the pretrained model.
2. **Learnable scalar gate.** A single learnable bias $b_g = {{MOE_INIT_GATE}}$ 
   gives $\sigma(b_g) \approx {{MOE_GATE_VALUE}}$ at initialization — 
   providing a smooth ramp-up.
3. **Architecture portability.** Weight transfer from DistilBERT, RoBERTa, 
   GPT-2, and Qwen3.5, with GQA, RoPE, nested encoders, and cross-attention.

On SST-2, ExpertChoiceMoEAttention achieves **{{D_ACC}}%** on DistilBERT 
(baseline {{BASE_D}}%), **{{R_ACC}}%** on RoBERTa (baseline {{BASE_R}}%), 
and **{{G_ACC}}%** on GPT-2 (baseline {{BASE_G}}%).

**We are explicit that this module increases parameter count** by roughly 
**{{D_PARAM_PCT}}%** per layer due to the $E$-expert MoE block and LoRA 
adapters. Its benefit is architectural: **guaranteed load balancing** and 
**dynamic capacity allocation**, which we show improve training stability 
relative to token-choice MoE.

**Keywords:** mixture of experts, expert-choice routing, load balancing, 
attention augmentation, multi-architecture transformers

---

## Lay Summary

Mixture-of-Experts (MoE) lets a neural network choose between several 
"sub-networks" per input, giving it more capacity without always using all 
parameters. In the standard MoE, each input picks a few experts — but this 
often leads to a few experts doing all the work while others stay idle. We 
flip the roles: each **expert** picks a fixed number of inputs. This 
guarantees every expert gets exactly the same workload and lets important 
tokens receive extra capacity. We apply this inside the attention block, 
after the standard attention computation, with an initial "mute" setting 
so the pretrained model is not disturbed at the start.

---

## Table of Contents

1. [Introduction](01_intro.md)
2. [Related Work](02_related.md)
3. [Background](03_background.md)
4. [Method](04_method.md)
5. [Architecture Integration](05_architecture.md)
6. [Experimental Setup](06_setup.md)
7. [Results](07_results.md)
8. [Analysis](08_analysis.md)
9. [Discussion](09_discussion.md)
10. [Limitations](10_limitations.md)
11. [Conclusion](11_conclusion.md)
12. [Reproducibility](12_reproducibility.md)
13. [References](13_references.md)

**Appendices:**
- [A. Mathematical Derivations](A_math.md)
- [B. Implementation Details](B_implementation.md)
- [C. Full Experimental Data](C_data.md)
- [D. Negative Results](D_negative_results.md)
- [E. Ethical Considerations](E_ethics.md)

---
