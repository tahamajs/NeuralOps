<!--
FILE: 00_meta.md
PURPOSE: Title page, metadata, abstract, lay summary
-->

# {{FULL_NAME}} ({{MODULE_NAME}}): {{SUBTITLE}}

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}} ({{CHANGELOG}})
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/{{MODULE_PATH}}
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Transformer self-attention scales quadratically with sequence length, making long-context inference expensive in memory-constrained deployment settings. Existing solutions based on sparse patterns, low-rank approximations, and KV-cache pruning trade off accuracy, stability, or implementation complexity. We introduce **{{FULL_NAME}}** (**{{MODULE_NAME}}**), a {{CATEGORY}} that {{CORE_BEHAVIOR_IN_ONE_SENTENCE}}.

{{MODULE_NAME}} combines {{COMPONENT_1}}, {{COMPONENT_2}}, and {{COMPONENT_3}} without retraining the base model. We transfer pretrained attention weights into the modified module and initialize trainable adapters to zero so the baseline behavior is preserved before recovery training. We evaluate on {{N_MODELS}} backbones on the {{TASK}} task with {{RECOVERY_STEPS}} LoRA recovery steps at rank {{LORA_RANK}}.

The method achieves **{{ACC_D}}%** on DistilBERT, **{{ACC_R}}%** on RoBERTa-base, and **{{ACC_G}}%** on GPT-2, relative to baselines of **{{BASE_D}}%**, **{{BASE_R}}%**, and **{{BASE_G}}%**, respectively. Parameter counts change by **{{PARAM_CHANGE_PCT}}%** ({{PARAM_CHANGE_NOTE}}), while peak inference VRAM decreases by **{{VRAM_CHANGE_PCT}}%** and wall-clock latency changes by **{{LATENCY_CHANGE}}×** due to {{LATENCY_REASON_SHORT}}.

Our results suggest that {{WHAT_WORKS}} and that {{WHAT_FAILS}} under {{FAILURE_CONDITION}}. We release the implementation, evaluation scripts, and checkpoints to support reproducibility.

**Keywords:** efficient attention, {{KEYWORD_2}}, {{KEYWORD_3}}, {{KEYWORD_4}}, transformer compression

---

## Lay Summary

{{LAY_SUMMARY}}

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
