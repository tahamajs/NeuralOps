<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# AASVD: Activation-Aware SVD with Streaming EMA Covariance for Transformer Compression

<!-- 
TITLE NOTES:
- "AASVD" = Activation-Aware SVD
- Avoid "efficient" or "optimal" without evidence
- Alternative: "AASVD: Streaming Covariance-Weighted Low-Rank Factorization for Transformers"
-->

**Authors:**
- Your Name¹ — you@university.edu
- Advisor Name² — advisor@university.edu

**Affiliations:**
- ¹ University Name, Department of Computer Science
- ² 0.85, 0.85

**Corresponding author:** Your Name
**Preprint date:** 2026-09-15
**Version:** v1
**Code:** https://github.com/your-org/SVD_Project/tree/main/framework/compression_engine/aasvd_compressor.py
**Model weights:** https://huggingface.co/Hooshaai
**License:** MIT

---

## Abstract

Low-rank factorization of transformer weight matrices is a classical approach 
to reducing model size and inference memory. However, standard truncated SVD 
minimizes unweighted Frobenius error, treating every input dimension as equally 
important. Transformer activations exhibit both **heterogeneous variance** 
(some features are far more active than others) and **strong inter-feature 
correlations**, both of which standard SVD ignores.

We introduce **AASVD** (**A**ctivation-**A**ware **S**VD), which addresses this 
by weighting the factorization through the empirical input activation 
covariance matrix $C_X = \mathbb{E}[x x^\top]$. We accumulate $C_X$ using a 
streaming Exponential Moving Average (EMA) over calibration batches, requiring 
only $\mathcal{O}(d^2)$ CPU memory regardless of calibration set size. Each 
weight $W$ is transformed as $W_\text{weighted} = W C_X^{1/2}$, factorized via 
SVD, and the right factor is unweighted by $C_X^{-1/2}$ to recover the original 
parameter space.

AASVD replaces each `nn.Linear` with a structural `SVDLinear` module computing 
$y = (x V^\top) U^\top + b$, achieving a genuine parameter reduction from 
$d_\text{in} d_\text{out}$ to $r(d_\text{in} + d_\text{out})$. AASVD is the only method achieving genuine compression at ranks $r \le 128$, where diagonal methods fail. On DistilBERT, 
RoBERTa-base, and GPT-2 with rank $r = 288$, AASVD achieves 
**4.0×–3.5×** compression ratios with accuracy drops of only 
**-1.26%–+1.37%** on SST-2 after 50 LoRA recovery 
steps. Peak inference VRAM decreases by **40%**. We provide a 
production-hardened implementation with Frobenius validation, automatic 
per-layer fallback, GPT-2 Conv1D support, and a bundled multi-domain 
calibration corpus.

**Keywords:** activation-aware compression, streaming covariance, EMA, low-rank 
factorization, transformer efficiency

---

## Lay Summary

Neural networks contain millions of numbers ("weights") that transform input 
data. Many of these weights act on features that rarely change during real 
usage, so they can be approximated without losing accuracy. AASVD identifies 
which input features matter most by running the model on a small set of 
example sentences and recording how active each feature becomes. It then 
uses this information to compress the weights — removing redundant directions 
while preserving important ones. On three popular language models, AASVD 
removes 60–80% of the parameters with only 1–3% loss in accuracy. This 
makes models easier to store, load, and run on modest hardware.

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
