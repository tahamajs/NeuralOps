<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Progressive Token Pruning: Learnable Token Selection for Efficient Transformer Attention

<!-- 
TITLE NOTES:
- "Progressive" emphasizes per-layer independent scoring
- "Learnable" emphasizes the scoring network
- Alternative: "PTP: Learnable Token Selection with Guaranteed [CLS] Retention"
-->

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/progressive_token_pruning.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Transformer inference scales quadratically with sequence length, but many 
tokens are redundant for downstream tasks. Prior work on token pruning 
either uses fixed heuristics (attention scores, magnitude) or requires 
retraining the full model. Both approaches have limitations: fixed heuristics 
may not align with task-specific importance, and full retraining is 
expensive.

We present **Progressive Token Pruning** (PTP), a learnable token-selection 
module that scores each token with a small neural network and retains the 
top-$k$ fraction. The module preserves the **[CLS]** token unconditionally, 
handles **padding** explicitly, and supports **causal masking** by preventing 
queries from attending to future keys even after pruning. Crucially, PTP 
is a **drop-in replacement** for standard attention: pretrained Q/K/V/O 
projections are copied exactly, and only the score network plus LoRA adapters 
are trainable.

We evaluate PTP on DistilBERT, RoBERTa-base, and GPT-2 on SST-2 with 
$k/L = 0.5$. PTP achieves **86.4%**, **78.2%**, and 
**77.5%** accuracy, within **0.85%**, **0.85%**, and 
**0.85%** of the dense baselines. Attention memory is reduced by 
**32×** and inference latency by **2.3 s×** at 
$L = 0.85$. The learned scores reveal systematic token importance: 
the [CLS] token and content words dominate, while function words and 
duplicate subword tokens are pruned.

**Keywords:** token pruning, learnable selection, efficient transformer, 
attention, sequence compression

---

## Lay Summary

Transformer models process every token in a sentence, but not all tokens 
matter equally. In sentiment classification, for example, the words "good" 
or "terrible" carry far more weight than "the" or "a."

Our method, Progressive Token Pruning (PTP), learns which tokens to keep and 
which to discard. A small neural network scores each token based on its 
hidden representation, and only the highest-scoring tokens are used in the 
attention computation. The [CLS] token — which carries the classification 
signal — is always kept. Padding tokens are always discarded.

Unlike earlier pruning methods, PTP is fully learnable and integrates with 
pretrained models without retraining. On three benchmark models, it 
achieves near-baseline accuracy while cutting attention computation 
substantially.

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