# VSpec++: Variational Spectral Token-Adaptive Attention

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** Your Name
**Preprint date:** 2026-09-16
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/attentions/vspec_attention.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

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
50 recovery steps. VSpec++ achieves **87.5%** on 
DistilBERT (baseline 90.71%), **75.0%** on RoBERTa (baseline 
78.56%), and **77.0%** on GPT-2 (baseline 79.36%). Attention 
parameters reduce by **2.2×–2.2×**, and peak VRAM decreases 
by **15%**.

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

---