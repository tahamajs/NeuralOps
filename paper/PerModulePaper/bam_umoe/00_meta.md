# BAM-UMoE: Unified Mixture of Experts Attention with Step-0 Representation Preservation

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/bam_umoe.py
**License:** {{LICENSE}}

---

## Abstract

Mixture of Experts (MoE) layers increase transformer capacity without 
proportional compute increases, but applying MoE to **attention value 
refinement** post-hoc is non-trivial: random initialization perturbs the 
pretrained representation, destabilizing recovery. We introduce **BAM-UMoE** 
(**B**idirectional **A**ttention **M**odulation with **U**nified **M**ixture 
**of E**xperts), which appends a top-2 MoE branch to the value pathway of 
each attention block with a **three-part step-0 preservation mechanism**:

1. **Zero-initialized final expert layers.** Every expert's output projection 
   starts at exactly zero, so the MoE branch contributes nothing at step 0.
2. **Negative gating bias.** A learnable gate with bias initialized to $-3.0$ 
   produces $\sigma(-3) \approx 0.047$, gently ramping up MoE contribution.
3. **Frozen base attention.** The original $W_Q, W_K, W_V, W_O$ projections 
   are copied exactly and frozen; only LoRA adapters and the MoE branch train.

The combination guarantees that **the initial forward pass is mathematically 
identical to the pretrained model**, allowing MoE capacity to be added without 
catastrophic drift.

BAM-UMoE supports flat (DistilBERT), nested (RoBERTa/BERT), causal (GPT-2), 
and GQA+RoPE (Qwen/LLaMA) architectures without modification. It additionally 
supports **cross-attention** for encoder-decoder models, and handles Conv1D 
fusions, nested encoders, and padding-aware masking natively.

**Important disclosure.** BAM-UMoE is **not** a compression method. Adding 
8 experts per layer increases parameter count substantially (see Table 7.2); 
we report all costs transparently. The method's value lies in **capacity 
expansion** rather than size reduction.

**Keywords:** mixture of experts, attention refinement, step-0 preservation, 
LoRA, multi-architecture

---

## Lay Summary

Mixture of Experts (MoE) is a technique where multiple small "expert" networks 
process the same input, and a router decides which experts to use for each 
token. This increases model capacity without proportionally increasing 
compute. Applying MoE to a **pretrained** transformer is hard: random 
initialization perturbs the model and destroys its pretrained knowledge. 
BAM-UMoE solves this by carefully initializing the MoE branch so that at 
training step 0, the model behaves exactly like the pretrained model. Then, 
during fine-tuning, the MoE branch gradually "wakes up" and learns to add 
value. The trade-off is a substantial increase in model size — this is a 
capacity expansion method, not a compression method.

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