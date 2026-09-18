<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# HALO-ReGLA: Hybrid Dense-Linear Attention with Rectified Gated Linear Attention

<!-- 
TITLE NOTES:
- "HALO" = Hardware-Aware Linear Optics
- "ReGLA" = Rectified Gated Linear Attention
- Emphasize: hybrid, dense+linear, learnable gate
- Alternative: "HALO-ReGLA: A Hybrid Attention Mechanism for Long-Context Inference"
-->

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/halo_regla.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Transformer attention at inference is bound by two opposing constraints: dense 
softmax attention delivers high-quality attention patterns but scales 
quadratically with sequence length, while linear attention offers linear-time 
inference at the cost of expressiveness. Existing hybrid methods either use 
fixed schedules (alternating layers) or require training from scratch, 
preventing direct adoption of pretrained weights.

We introduce **HALO-ReGLA**, a hybrid attention module that runs a **dense 
softmax stream** and a **rectified linear stream** in parallel within each 
layer, and combines them with a **per-head learnable gate** initialized to 
favor the dense branch. The linear stream uses the rectified kernel 
$\phi(x) = \mathrm{elu}(x) + 1$, whose non-negativity enables numerically 
stable cumulative-sum formulation for both causal and bidirectional attention. 
The gate is initialized with bias 1.8 so the model starts close to the dense 
behavior and learns to shift toward the linear branch when the task allows.

We evaluate HALO-ReGLA on DistilBERT, RoBERTa-base, and GPT-2 on SST-2. With 
LoRA recovery at rank 16, HALO-ReGLA achieves **89.5%**, 
**76.5%**, and **77.5%** accuracy, within **-1.2%**, 
**-2.1%**, and **-1.9%** of the dense baselines. Inference 
latency is 1.5×–1.3× relative to dense attention, 
with attention memory reduced by **2×** at long sequences. 
We provide a production-hardened PyTorch implementation supporting flat, 
nested, and causal architectures with RoPE, GQA, and safe fallback.

**Keywords:** hybrid attention, linear attention, learnable gating, rectified 
kernel, efficient transformer

---

## Lay Summary

Attention in transformers has a fundamental trade-off. Standard attention 
looks at every pair of tokens — high quality but slow for long sequences. 
Linear attention summarizes past tokens into a fixed-size state — fast but 
less precise.

HALO-ReGLA runs both at once: a full attention path and a linear summary 
path. A small trainable "gate" per attention head decides how much to trust 
each path. Initially, the gate trusts the full path; during training, it 
learns to shift toward the linear path when the task allows.

We test on three language models. HALO-ReGLA retains most of the original 
accuracy while reducing memory and improving scalability. The learned gates 
reveal which heads truly need full attention and which can be approximated 
linearly — a useful signal for model analysis.

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
