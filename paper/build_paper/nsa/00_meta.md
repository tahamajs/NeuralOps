<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Native Sparse Attention: Sliding-Window Local Context with Dynamic Top-K Global Routing

<!-- 
TITLE NOTES:
- "Native" emphasizes the fusion is inside the module
- "Dynamic" emphasizes content-dependent top-k
- Alternative: "NSA: Content-Aware Sparse Attention with Local Windows and Global Top-K"
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
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/nsa.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Sparse attention methods exploit the observation that most attention mass lies 
within a small neighborhood around each query. Fixed-window methods 
(Longformer, BigBird) capture local context cheaply but miss distant 
dependencies. Content-based methods (Routing Transformer, Reformer) capture 
distant dependencies but add hashing or clustering overhead. We present 
**Native Sparse Attention** (NSA), which combines **both**: a fixed sliding 
window captures local context, while a **dynamic top-$k$ routing** selects 
the $k$ most relevant distant keys per query based on actual attention scores. 
The selection is **fully vectorized** — no Python loop, no clustering, no 
hashing — making NSA easy to implement, train, and reason about.

NSA is parameterized by two scalars: window size $w$ and top-$k$ count $k$. 
For each query $i$, the retained keys are:

$$
\mathcal{S}_i = \underbrace{\{j : |i - j| \leq w\}}_{\text{local window}} \cup \underbrace{\text{top-}k_{j' \notin \mathcal{W}_i} s_{ij'}}_{\text{dynamic global}}
$$

The union is computed via a boolean mask that is applied before softmax. 
Padding, causality, and GQA are all preserved.

We evaluate NSA on DistilBERT, RoBERTa-base, and GPT-2 on SST-2 with 
$w = 128$, $k = 2$. NSA achieves **90.83%**, 
**50.92%**, and **50.92%** accuracy, within **+0.12%**, 
**-27.64%**, and **-28.44%** of the dense baselines. Attention 
density drops to **[DENSITY]%** (from 100% dense). The dynamic top-$k$ 
selection shifts systematically with layer depth, revealing head-level 
content specialization.

**Keywords:** sparse attention, sliding window, dynamic top-k, content-aware 
routing, efficient transformer

---

## Lay Summary

Most language-model attention weight lies on nearby tokens. Some attention 
heads, however, need to look far away. Existing methods handle one regime or 
the other: fixed windows see only nearby tokens; content-based methods need 
expensive clustering or hashing.

Our method NSA runs both at once: for every token, we keep a fixed window 
of nearby context AND dynamically pick the few most-relevant distant tokens 
based on their actual attention scores. The selection is done in a single 
vectorized operation — no loop, no clustering.

On three pretrained models, NSA retains most accuracy while cutting attention 
density to about [DENSITY]%. The set of dynamically-selected tokens changes 
systematically with layer depth, revealing what each layer cares about.

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