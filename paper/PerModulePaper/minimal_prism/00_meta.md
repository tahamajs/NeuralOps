<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# MinimalPRISM: Spectral Energy-Adaptive Low-Rank Factorization for Efficient Transformer Attention

<!-- 
TITLE NOTES:
- "MinimalPRISM" = Parameter Reduced InjectIve Spectral Module (or just PRISM)
- "Spectral Energy-Adaptive" highlights the rank-selection novelty
- Alternative: "MinimalPRISM: Singular-Spectrum Energy-Factorized Attention for Multi-Architecture Transformers"
-->

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}} ({{CHANGELOG}})
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/minimal_prism.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

We present **MinimalPRISM** (**P**arameter-**R**educed **I**njective 
**S**pectral **M**odule), a drop-in replacement for the four attention 
projections of standard transformer layers. Each projection 
$W \in \mathbb{R}^{d_\text{out} \times d_\text{in}}$ is factorized as 
$W \approx U V$ via truncated SVD, with $U \in \mathbb{R}^{d_\text{out} \times r}$ 
and $V \in \mathbb{R}^{r \times d_\text{in}}$. This reduces parameter count 
from $d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$, achieving 
a **{{D_RATIO}}×** reduction per projection on DistilBERT at rank 
$r = {{RANK}}$.

Our central contribution is **spectral energy-adaptive rank selection**. Instead 
of fixing $r$ uniformly, MinimalPRISM computes the cumulative spectral energy 
$E(r) = \sum_{i \le r} \sigma_i^2 / \sum_i \sigma_i^2$ of each weight matrix 
and selects the smallest rank such that $E(r) \ge \eta$, with $\eta = {{ENERGY}}$ 
by default. This automatically adapts per-layer rank based on spectral decay, 
concentrating compression where the spectrum is already concentrated.

We provide a production-hardened implementation supporting four transformer 
families — DistilBERT (flat), RoBERTa (nested), GPT-2 (causal with Conv1D), 
and Qwen3.5 (GQA + RoPE) — with:
(i) explicit **nested encoder detection at injection time** for correctness,
(ii) Conv1D weight transposition for GPT-2,
(iii) GQA support via `repeat_kv`,
(iv) strict Frobenius norm validation on extracted weights,
(v) LoRA adapters initialized to zero for recovery fine-tuning,
(vi) safe abort with pre-test on the first layer.

On DistilBERT, RoBERTa-base, and GPT-2 with rank cap $r = {{RANK}}$ and 
energy $\eta = {{ENERGY}}$, MinimalPRISM achieves **{{D_RATIO}}×–{{G_RATIO}}×** 
parameter reduction with accuracy drops of only **{{D_DELTA}}%–{{G_DELTA}}%** 
after {{RECOVERY_STEPS}} LoRA recovery steps. Peak inference VRAM decreases 
by **{{VRAM_REDUCTION}}%**.

**Keywords:** low-rank factorization, spectral energy, adaptive rank, SVD, 
attention compression, multi-architecture transformers, LoRA recovery

---

## Lay Summary

Modern language models spend most of their parameters on four large matrices 
inside each attention layer. MinimalPRISM replaces each matrix with a smaller 
pair whose product approximates the original. Rather than fixing the size of 
the smaller pair, MinimalPRISM measures how concentrated the matrix is — if a 
few directions dominate, those are kept; if the matrix is uniform, more 
directions are preserved. On three popular models (DistilBERT, RoBERTa, GPT-2), 
MinimalPRISM removes 60–75% of the attention parameters with only 1–3% loss 
in accuracy. The method requires no calibration data and works across four 
different transformer architectures.

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