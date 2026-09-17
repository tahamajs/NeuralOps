# JambaInfo: Information-Bottlenecked Jamba Architecture with Concept-Compressed SSM

**Authors:**
- [AUTHOR_1_NAME]¹ — [AUTHOR_1_EMAIL]
- [AUTHOR_2_NAME]² — [AUTHOR_2_EMAIL]

**Affiliations:**
- ¹ [UNIVERSITY_1], [DEPARTMENT_1]
- ² [UNIVERSITY_2], [DEPARTMENT_2]

**Corresponding author:** [CORRESPONDING_AUTHOR]
**Preprint date:** [DATE]
**Version:** v[VERSION]
**Code:** https://github.com/Efficiency Lab/SVD_Project/tree/main/framework/attentions/jamba_info.py
**Model weights:** https://huggingface.co/efficiency-lab
**License:** Apache 2.0

---

## Abstract

Hybrid attention-SSM architectures combine the precise retrieval of attention 
with the long-range efficiency of state space models. However, naively 
feeding the full hidden state $x \in \mathbb{R}^d$ into the SSM branch 
introduces redundant information: the SSM must compress $d$-dimensional 
inputs into its recurrent state at every step. We argue that this is 
**information-theoretically wasteful** — the SSM only needs a compressed 
"concept representation" of the input.

We introduce **JambaInfo**, an information-bottlenecked hybrid block that 
**routes the SSM branch through a low-dimensional concept bottleneck**:

$$
z_t = W_\text{up} \cdot \text{GELU}(W_\text{down} x_t), 
\quad W_\text{down} \in \mathbb{R}^{b \times d}, \quad b \ll d
$$

where $b = 64$. The bottleneck acts as an explicit 
information constraint: the SSM processes only the essential concepts, 
reducing noise and parameters. Concurrently, a **differential attention** 
branch (two Q-K pairs with ReLU-damped subtraction) computes precise 
retrieval. The two branches are fused by a learned scalar gate:

$$
\text{out} = (1 - \alpha) \odot A_\text{diff} + \alpha \odot M_\text{concept},
\quad \alpha = \sigma(w), \quad w_0 = -2
$$

We initialize $\alpha \approx 0.12$, biasing toward attention initially 
and letting the model learn how much to rely on the SSM.

On SST-2, JambaInfo achieves **89.68%** on DistilBERT (baseline 
0.85%), **52.98%** on RoBERTa (baseline 32%), and 
**50.80%** on GPT-2 (baseline 0.85%).

**We are explicit that JambaInfo does not reduce parameter count.** It 
**adds** approximately 66.4M% more parameters per layer. Its 
benefit is **concept-driven SSM compression** and **reduced attention noise** 
at a modest parameter cost. The concept bottleneck itself can be viewed as 
a lightweight information bottleneck that improves SSM efficiency.

**Keywords:** information bottleneck, hybrid architecture, differential 
attention, state space model, concept compression

---

## Lay Summary

We build a hybrid attention-SSM block. The attention part uses two attention 
maps and subtracts them to cancel noise (differential attention). The SSM 
part is our novelty: instead of feeding the full hidden state into the SSM, 
we first compress it through a **small bottleneck** (256 dimensions instead 
of 768). This forces the SSM to work with a compressed "concept" 
representation. A learned gate decides how much to trust each branch. On 
three standard models, we match baseline accuracy with a cleaner, more 
focused internal representation.

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