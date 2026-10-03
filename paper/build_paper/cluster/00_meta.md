# ClusterFormer: Token Clustering Attention with Differentiable K-Means and Local Sliding Window

**Authors:**
- Your Name¹ — you@university.edu
- 0.85² — 0.85

**Affiliations:**
- ¹ University, CS
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/cluster_attention.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Standard self-attention has quadratic complexity in sequence length, 
$\mathcal{O}(L^2 d)$, which dominates the cost of transformer inference 
beyond a few hundred tokens. Existing linear attention methods achieve 
$\mathcal{O}(L d)$ complexity but sacrifice representational capacity by 
either restricting attention to local windows (Longformer) or approximating 
softmax via random features (Performer). We introduce **ClusterFormer**, a 
**differentiable K-means clustering attention** that groups queries into 
$K$ semantic cluster centroids, evaluates **centroid-to-key cross-attention**, 
and returns to individual queries via a soft assignment. This yields 
$\mathcal{O}(K L d)$ complexity while preserving **global context**, 
unlike local-only methods.

ClusterFormer combines four components:

1. **Differentiable K-means** over query representations with padding-aware 
   initialization and 3 Lloyd iterations.
2. **Centroid-to-key global cross-attention** producing $K$ context vectors 
   at cost $\mathcal{O}(K L d)$.
3. **Learnable soft assignment** with a temperature parameter to redistribute 
   centroid contexts back to individual queries.
4. **Local sliding window residual** blended with the global branch via a 
   learnable gate, capturing nearby syntactic structure that clustering may 
   miss.

Base Q/K/V/O projections are frozen and copied from the pretrained model; 
LoRA rank-16 adapters and learnable temperature/blend scalars provide the 
only trainable capacity. On DistilBERT, RoBERTa-base, and GPT-2 with SST-2 
as the evaluation task, ClusterFormer maintains accuracy within 
**-0.69%–+8.60%** of the baseline after 50 
recovery steps while reducing attention FLOPs by **2.7×** at 
sequence length $L = 128$ with $K = 16$ clusters.

**Keywords:** linear attention, differentiable clustering, token routing, 
sliding window, efficient transformers

---

## Lay Summary

Standard transformer attention compares every token to every other token, 
which becomes slow when the sequence gets long. ClusterFormer takes a 
different approach: it groups similar tokens into a small number of 
"clusters" (like sorting people by hometown at a conference) and then 
lets each cluster attend to all tokens at once. This is much faster, and 
it still keeps global context — every cluster sees the whole sequence. 
To capture fine-grained local structure, we also add a small sliding window 
that attends to nearby tokens. The two signals are mixed together with a 
learnable gate. On three popular language models, ClusterFormer matches 
the original models' accuracy on sentiment analysis while using a fraction 
of the attention computation.

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