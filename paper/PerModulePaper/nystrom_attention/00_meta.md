<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 00_meta.md
     PURPOSE: Title, authors, abstract, keywords
     DEPENDENCIES: None
     ═══════════════════════════════════════════════════════════════════════════ -->

# Nyströmformer on Non-BERT Architectures: A Portability and Reproduction Study

<!-- 
TITLE NOTES:
- This is a REPRODUCTION paper, not a novel method
- Original paper: Xiong et al. 2021 (AAAI)
- Emphasis on: multi-architecture porting, production hardening
- Alternative: "Nyströmformer for Multi-Architecture Transformers: Portability and Production Hardening"
-->

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION] ([CHANGELOG])
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/nystrom.py
**Original paper:** https://arxiv.org/abs/2102.03902
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

**Nyströmformer** (Xiong et al., 2021) approximates standard softmax attention 
via the Nyström low-rank kernel method. Instead of materializing the full 
$N \times N$ attention matrix, it selects $m \ll N$ landmark tokens and 
computes three smaller attention matrices, achieving $\mathcal{O}(N m d)$ 
complexity instead of $\mathcal{O}(N^2 d)$. Landmarks are chosen via 
**adaptive average pooling**, and a **pseudoinverse** of the landmark attention 
matrix is computed iteratively (Newton-Schulz).

We present a **reproduction** of Nyströmformer, ported to four transformer 
families that the original paper does not target: DistilBERT (flat), 
RoBERTa/BERT (nested), GPT-2 (causal Conv1D), and Qwen3.5 (GQA + RoPE). Our 
implementation:
(i) preserves exact softmax attention for short sequences ($N \leq 128$) and 
for causal decoders,
(ii) uses an iterative pinv with a fallback to `torch.linalg.pinv`,
(iii) handles Conv1D transposition, nested encoder bypass, GQA expansion, 
and RoPE,
(iv) attaches LoRA adapters for recovery fine-tuning.

We emphasize: **this is a reproduction and portability study, not a novel 
architecture.** Nyströmformer does **not** reduce parameter count or 
KV-cache size. It reduces **attention compute** for long sequences.

On SST-2 (sequence length 128), the Nyström branch does not activate (dense 
fallback), and Nyströmformer matches the baseline. On longer sequences, the 
Nyström approximation provides asymptotic savings.

**Keywords:** Nyströmformer, Nyström method, low-rank attention, kernel 
approximation, iterative pseudoinverse, Newton-Schulz, multi-architecture 
transformers, reproduction study

---

## Lay Summary

Standard transformer attention compares every token to every other token, 
which is expensive for long sequences. Nyströmformer, introduced by Xiong 
et al. (2021), instead compares tokens to a small set of "landmark" tokens, 
reducing the cost. This paper reproduces Nyströmformer and ports it to four 
model families that the original paper did not target. We confirm the 
original method works on these models and document the engineering required 
for the port.

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