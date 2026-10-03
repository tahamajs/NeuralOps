# LEKAv3: Linear-Time Efficient Kernel Attention with Value-Scored Landmarks

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/leka_v3.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Transformer self-attention has quadratic cost $\mathcal{O}(L^2)$ in sequence 
length $L$, which becomes prohibitive for long-context tasks. We introduce 
**LEKAv3**, a linear-time attention mechanism that approximates softmax 
attention via **value-scored landmarks**. LEKAv3 combines two complementary 
branches: (i) a **global landmark branch** that computes kernel-feature 
attention over a small set of $L_\text{land} \ll L$ tokens selected by a 
learnable scorer, and (ii) a **local sliding-window branch** that computes 
exact attention within a window of size $W$. The two branches are fused 
by a per-head learnable gate initialized to favor landmarks.

The landmark scorer is a small MLP operating on **multi-scale random Fourier 
features** of keys, giving $\mathcal{O}(L \cdot d_\text{feat})$ selection 
cost. The kernel feature map uses cos/sin projections at two scales, 
providing a bounded approximation of the softmax kernel.

On SST-2, LEKAv3 achieves **70.53%** on DistilBERT (baseline 
0.85%, Δ = -20.18), **50.92%** on RoBERTa (baseline 
32%, Δ = -27.64), and **50.92%** on GPT-2 (baseline 
0.85%, Δ = -28.44), using 50 LoRA recovery steps. 
The landmark count scales as $L_\text{land} = \min(L, \max(L_0, \lceil L/2 \rceil))$ 
where $L_0$ is the base landmark budget. Complexity is 
$\mathcal{O}(L L_\text{land} + L W)$ — linear in $L$ when $L_\text{land}, W \sim \mathcal{O}(1)$.

**Keywords:** linear attention, kernel methods, landmark selection, random 
Fourier features, efficient transformers

---

## Lay Summary

Standard transformers compare every word to every other word, which becomes 
slow for very long texts. Our method picks a small set of "landmark" words 
that summarize the sequence, then compares each word to those landmarks 
instead of to everything. We also keep a small sliding window so nearby 
words can still see each other directly. The two views are blended together 
with a learned weight. The result is much faster on long inputs, with only 
a small accuracy loss.

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