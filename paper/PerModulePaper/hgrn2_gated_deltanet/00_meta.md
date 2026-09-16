# HGRN2-GatedDeltaNet: A Hybrid Linear Recurrent Attention Module for Efficient Long-Context Transformers

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/hgrn2_gated_deltanet.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Transformer self-attention scales quadratically with sequence length, making 
long-context inference prohibitively expensive. Recent work has explored 
**linear recurrent alternatives** — HGRN, GLA, Mamba, DeltaNet — each with 
distinct trade-offs in expressivity, training stability, and hardware 
efficiency. We introduce **HGRN2-GatedDeltaNet**, a hybrid module that 
combines two complementary linear recurrence mechanisms: (i) **HGRN-2**, an 
element-wise recurrence with bounded forget gates, and (ii) **Gated DeltaNet**, 
a multi-head associative matrix memory with Delta-rule error updates. The 
two branches are fused by a **learnable token-level gate**:

$$
Y = \gamma \odot Y_\text{HGRN} + (1 - \gamma) \odot Y_\text{DeltaNet}, \quad
\gamma = \sigma(W_g X)
$$

Both branches run in $\mathcal{O}(L)$ time and memory, replacing the 
$\mathcal{O}(L^2)$ of standard attention. Base weights are transferred from 
the pretrained attention via a robust multi-architecture `copy_weights` 
procedure; LoRA adapters provide recovery capacity.

We evaluate on DistilBERT, RoBERTa-base, and GPT-2 on SST-2 with 
{{RECOVERY_STEPS}} LoRA recovery steps. HGRN2-GatedDeltaNet achieves 
**{{D_ACC}}%** on DistilBERT (baseline {{BASE_D}}%), **{{R_ACC}}%** on 
RoBERTa (baseline {{BASE_R}}%), and **{{G_ACC}}%** on GPT-2 (baseline 
{{BASE_G}}%).

**Crucially, we are explicit about the trade-offs:** the module **does not 
reduce parameter count** — in fact, it *increases* parameters by 
**{{D_PARAM_PCT}}%** due to LoRA and the gate. Additionally, the current 
implementation **runs {{SLOWDOWN}}× slower than baseline** on short 
sequences (L = 128) due to a Python-level sequential recurrence. The 
advantage of HGRN2-GatedDeltaNet appears only at long sequence lengths 
($L \geq {{BREAKEVEN}}$), where the $\mathcal{O}(L)$ scaling dominates.

**Keywords:** linear attention, recurrent neural networks, HGRN, DeltaNet, 
long-context efficiency, state-space models

---

## Lay Summary

Standard transformer attention compares every word to every other word, 
which gets very slow for long texts. We propose a hybrid module that instead 
processes text one word at a time, keeping a compressed memory of what 
came before. It combines two different memory mechanisms: one that 
remembers single numbers per feature, and one that remembers small matrices. 
A learned gate decides how much to trust each. Because both mechanisms 
process text sequentially, they run in time proportional to the text length, 
not its square. This makes them promising for very long documents — but on 
short sequences, they are slower than standard attention in their current 
form.

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