# TokenCompressedAttention: Learnable Token Selection for Efficient Transformer Inference

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/token_compressed.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

The quadratic cost of transformer attention grows with sequence length because 
every query attends to every key. For long-context inference, the KV cache 
becomes the dominant memory bottleneck. We introduce **TokenCompressedAttention**, 
a lightweight module that reduces the effective KV sequence length by selecting 
a subset of tokens based on a learnable importance score, while always 
preserving the special `[CLS]` token and respecting causal masking for 
decoders.

Our method adds a small scoring network (66.4M parameters) on 
top of a frozen pretrained attention. At compression ratio $\rho = 50\%$, 
we reduce the effective KV sequence length by $1 - \rho$, translating to 
**84.5%** lower attention compute and KV cache. On SST-2, 
TokenCompressedAttention achieves **50.92%** on DistilBERT (baseline 
0.85%), **50.92%** on RoBERTa (baseline 32%), and 
**50.92%** on GPT-2 (baseline 0.85%), using only 50 
LoRA recovery steps.

Crucially, we are explicit that this method **does not reduce parameter 
count** — the base weights are preserved and LoRA adapters are added. What 
it reduces is inference-time memory and compute. We provide a production-
hardened implementation supporting flat, nested, and causal architectures, 
with a built-in `[CLS]`-preservation guarantee and a numerically stable 
causal mask for decoder models.

**Keywords:** KV cache compression, token selection, efficient inference, 
learnable saliency, long-context transformers

---

## Lay Summary

When a transformer reads a long piece of text, it stores a representation 
of every word (a "key" and a "value") so later words can look back at it. 
For very long inputs, this storage grows quickly. Our method learns to 
identify which words are actually important and keeps only those, throwing 
away the rest. We always keep the first special `[CLS]` token, and we make 
sure that when a word is looking back, it only sees earlier words (not 
future ones). This lets the model run on longer inputs with less memory, 
at a small accuracy cost.

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