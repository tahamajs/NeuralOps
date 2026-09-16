# VSpec++: Variational Spectral Token-Adaptive Attention

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/vspec_attention.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Attention projection matrices in transformers are typically learned densely, 
yet their spectra are often highly non-uniform. We propose **VSpec++**, a 
variational spectral attention module that combines **three orthogonal 
gating mechanisms**:

1. **Token routing** — select top-$k$ tokens via a learned saliency network 
   with guaranteed `[CLS]` retention.
2. **Variational spectral gating** — modulate each truncated singular value 
   with a stochastic, input-dependent gate trained via the reparameterization 
   trick.
3. **Head gating** — weight attention heads per-example via a masked-pooled 
   network.

The Q/K/V/O projections are factorized via truncated SVD, initialized 
zero-shot from the pretrained weights, and augmented with LoRA adapters. 
All base weights (SVD factors) are frozen; only the gates, router, and LoRA 
adapters are trained.

We evaluate on DistilBERT, RoBERTa-base, and GPT-2 on SST-2 with 
{{RECOVERY_STEPS}} recovery steps. VSpec++ achieves **{{D_ACC}}%** on 
DistilBERT (baseline {{BASE_D}}%), **{{R_ACC}}%** on RoBERTa (baseline 
{{BASE_R}}%), and **{{G_ACC}}%** on GPT-2 (baseline {{BASE_G}}%). Attention 
parameters reduce by **{{D_RATIO}}×–{{G_RATIO}}×**, and peak VRAM decreases 
by **{{VRAM_REDUCTION}}%**.

The variational formulation provides a **Bayesian interpretation**: the 
spectral gate $g_i \sim q(g_i | x)$ is a posterior over per-singular-value 
importance, and the reparameterized sample $z = \mu + \sigma \epsilon$ 
allows gradient-based training of both mean and variance. At inference, we 
use $z = \mu$ (deterministic), giving a stable, non-stochastic attention 
computation.

**Keywords:** variational inference, spectral attention, token routing, 
low-rank factorization, LoRA

---

## Lay Summary

Transformer attention has four big projection matrices. We propose a 
method that (a) shrinks these matrices via SVD, (b) selects only the most 
important tokens for attention, and (c) learns to weight each singular 
value and each attention head. The spectral weighting is **probabilistic** 
during training — the model samples different weightings on each forward 
pass and learns from them. At inference, we use the average weighting, 
giving a stable and cheap forward pass. On three standard models, we 
reduce attention parameters while keeping accuracy close to baseline.

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
```

---