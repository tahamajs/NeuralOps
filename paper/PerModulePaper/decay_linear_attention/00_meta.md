# DecayLinearAttention: O(1)-Memory Linear Attention with Data-Dependent Exponential Decay

**Authors:**
- {{AUTHOR_1_NAME}}¹ — {{AUTHOR_1_EMAIL}}
- {{AUTHOR_2_NAME}}² — {{AUTHOR_2_EMAIL}}

**Affiliations:**
- ¹ {{UNIVERSITY_1}}, {{DEPARTMENT_1}}
- ² {{UNIVERSITY_2}}, {{DEPARTMENT_2}}

**Corresponding author:** {{CORRESPONDING_AUTHOR}}
**Preprint date:** {{DATE}}
**Version:** v{{VERSION}}
**Code:** https://github.com/{{ORG}}/SVD_Project/tree/main/framework/attentions/decay_linear_attention.py
**Model weights:** https://huggingface.co/{{HF_ORG}}
**License:** {{LICENSE}}

---

## Abstract

Linear attention replaces the quadratic softmax with an associative kernel, 
achieving $\mathcal{O}(L)$ complexity. However, two practical issues remain: 
(i) standard linear attention has **no position-dependent decay**, so all 
past keys contribute equally to the current context, and (ii) causal 
implementations often materialize an $\mathcal{O}(L)$ intermediate state, 
negating the memory benefit.

We introduce **DecayLinearAttention**, which addresses both via:

1. **Data-dependent exponential decay per head.** We learn a per-head decay 
   rate $\lambda_{h,t} \in (0, 1)$ from the input:
   
   $$
   \lambda_{h,t} = \sigma\!\left(W_g x_t + b_g + \ell_h\right)
   $$
   
   where $\ell_h \in [{{LOG_DECAY_MIN}}, {{LOG_DECAY_MAX}}]$ is a per-head 
   learned bias. Different heads learn different context horizons.
2. **Chunked scan with O(1) state.** We partition the sequence into chunks 
   of size $C = {{CHUNK_SIZE}}$ and carry a compact state 
   $S \in \mathbb{R}^{d \times d}$ between chunks. The intermediate state is 
   **$\mathcal{O}(1)$ in sequence length**, in contrast to the 
   $\mathcal{O}(L)$ intermediates used by cumsum-based causal linear 
   attention.
3. **ELU+1 feature map.** We use $\phi(x) = \text{elu}(x) + 1$, which is 
   positive and cheap.

We support **bidirectional** attention for encoders (forward and backward 
scans averaged) and **causal** for decoders, along with GQA, RoPE, and 
**cross-attention**.

On SST-2, DecayLinearAttention achieves **{{D_ACC}}%** on DistilBERT 
(baseline {{BASE_D}}%), **{{R_ACC}}%** on RoBERTa (baseline {{BASE_R}}%), 
and **{{G_ACC}}%** on GPT-2 (baseline {{BASE_G}}%). Peak inference VRAM is 
**{{D_VRAM}} MB** on DistilBERT (baseline {{BASE_VRAM_D}} MB), and the state 
memory is **constant** in $L$.

**Keywords:** linear attention, data-dependent decay, chunked scan, O(1) 
memory, cross-attention

---

## Lay Summary

Standard attention compares every word to every other word, which gets slow 
for long texts. A popular alternative is "linear attention" — it summarizes 
the past into a fixed-size state and updates it as new words arrive. But 
most linear attention treats all past words equally, which is wrong: recent 
words usually matter more. Our method learns a per-head decay rate that 
controls how quickly the past is forgotten. Old words decay exponentially; 
recent words dominate. We also implement the linear scan efficiently by 
processing the sequence in small chunks and carrying a fixed-size state 
between them, so memory usage does not grow with text length.

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
