<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Adaptive Sparse Attention: Learned Per-Head Sparsity for Efficient Transformer Inference

<!-- 
TITLE NOTES:
- "Adaptive" emphasizes learnable per-head
- "Learned" emphasizes the differentiable mechanism
- Alternative: "ASA: Differentiable Top-K Sparse Attention with Per-Head Budgets"
-->

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/adaptive_sparse_attention.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Dense softmax attention computes $L^2$ pairwise scores per head, but empirical 
studies have repeatedly shown that the attention distribution is highly 
concentrated: only a small fraction of keys receive most of the attention mass. 
This suggests that most of the $L^2$ scores can be pruned with minimal impact 
on quality. Static sparsity patterns (fixed windows, fixed strides) fail to 
capture this variability, and existing dynamic methods either ignore per-head 
heterogeneity or introduce non-differentiable top-$k$ operations that hinder 
end-to-end training.

We introduce **Adaptive Sparse Attention** (ASA), a per-head learnable 
sparsification mechanism. Each of the $H$ heads maintains a single scalar 
logit $\ell_h$ controlling its retention budget: $k_h = \lfloor (1 - \sigma(\ell_h)) \cdot L \rfloor$. 
The top-$k$ selection is made differentiable via a **soft sigmoid threshold**: 
after computing the top-$k$-th score $\tau_h$, we apply a soft mask 
$\sigma((s_i - \tau_h) / T)$ that concentrates mass on the retained entries.

We evaluate ASA on DistilBERT, RoBERTa-base, and GPT-2 on SST-2. At default 
sparsity {{SPARSITY}} on decoders and {{SPARSITY_ENC}} on encoders, ASA 
achieves **{{D_ACC}}%**, **{{R_ACC}}%**, and **{{G_ACC}}%** accuracy, within 
**{{D_DELTA}}%**, **{{R_DELTA}}%**, and **{{G_DELTA}}%** of the respective 
dense baselines. Attention FLOPs are reduced by up to **{{FLOP_REDUCTION}}×**. 
The learned per-head budgets show systematic structure: some heads converge 
to near-dense (sparsity ≈ 0) and others to near-sparse (sparsity ≈ 0.9), 
demonstrating head heterogeneity.

**Keywords:** sparse attention, learned sparsity, top-k, differentiable 
masking, per-head budgets

---

## Lay Summary

Attention is the core operation in transformers. For a sequence of 1000 
words, it computes a million pairwise scores — but in practice, most of these 
scores are near zero and contribute nothing. Throwing them away saves compute 
without hurting quality.

The challenge is *which* scores to throw away. A fixed rule (e.g., "keep only 
nearby tokens") works poorly when important context is distant. Our method 
lets each attention head **learn** its own rule. A single trainable number 
per head controls what fraction of keys that head attends to. The learning 
is fully differentiable, so it can be trained end-to-end alongside the model.

We test this on three popular language models and find that the learned 
budgets are highly non-uniform: some heads remain dense, others become almost 
completely sparse. This shows that not all heads are equally important — a 
fact that static sparsity methods ignore.

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