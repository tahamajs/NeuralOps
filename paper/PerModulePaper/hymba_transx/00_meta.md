# HymbaTransX: A Hybrid Differential Attention and State Space Model Block

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/hymba_transx.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Transformers and state space models (SSMs) have complementary strengths: 
attention captures precise pairwise dependencies but scales quadratically, 
while SSMs scale linearly but compress distant context. We introduce 
**HymbaTransX**, a hybrid block that combines **differential attention** 
(Ye et al., 2024) with a **SimpleSSM** branch via a learned fusion gate:

$$
\text{out} = \alpha \odot A_\text{diff} + (1 - \alpha) \odot M_\text{ssm},
\quad \alpha = \sigma(w)
$$

The differential attention branch maintains **two independent query-key 
pairs** $(Q_1, K_1)$ and $(Q_2, K_2)$ and computes a ReLU-damped difference:

$$
A_\text{diff} = \text{ReLU}\!\left(\text{softmax}(Q_1 K_1^\top) - \lambda \cdot \text{softmax}(Q_2 K_2^\top)\right)
$$

with $\lambda = 0.2$. This cancels common-mode noise in the attention 
pattern, improving signal-to-noise ratio. The SSM branch provides 
$O(L)$ long-range mixing. Both branches share input projections.

HymbaTransX supports GQA, causal masking for decoders, bidirectional 
recurrence for encoders, and both external RoPE (from the parent model) 
and internal `HeadRoPE` fallback. We evaluate on DistilBERT, RoBERTa, and 
GPT-2 on SST-2, achieving **49.08%** (DistilBERT, baseline 0.85%), 
**51.15%** (RoBERTa, baseline 32%), and **57.57%** 
(GPT-2, baseline 0.85%).

**We are explicit that HymbaTransX does not reduce parameter count.** In 
fact, the module **adds** approximately 66.4M% more parameters 
per layer. Its benefit is **attention-noise reduction** and **SSM-augmented 
long-range context**, at a small training and inference overhead.

**Keywords:** differential attention, hybrid architecture, state space model, 
SSM, attention noise cancellation

---

## Lay Summary

Standard attention sometimes gets distracted by irrelevant tokens. 
Differential attention fixes this by computing two attention patterns and 
subtracting one from the other, keeping only the parts that disagree. We 
combine this cleaner attention with a state space model — a compressed, 
recurrent memory of what came before. A learned gate decides how much to 
trust each. The result is a more focused attention block that also has 
long-range context, at a modest parameter cost.

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