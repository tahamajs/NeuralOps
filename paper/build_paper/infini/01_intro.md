<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Full self-attention is $\mathcal{O}(L^2)$ in sequence length $L$, which is 
prohibitive for long-context inference. **Sliding-window attention** reduces 
this to $\mathcal{O}(L \cdot W)$ with $W \ll L$, but at a fundamental cost: 
tokens outside the window are invisible to each other. Information from the 
beginning of a long document cannot reach tokens near the end.

**Compressive memory** offers one solution. Instead of keeping every past 
token's key-value pair, maintain a small learned matrix $M \in \mathbb{R}^{d_c \times d_h}$ 
per head that summarizes the global context. Queries attend to this memory 
matrix as if it were a set of $d_c$ learned "virtual tokens".

**Infini-Attention** combines both ideas: local sliding-window attention for 
precise, position-sensitive computation, plus a learned compressive memory 
for global context. The two branches are combined through a learnable scalar 
gate:

$$
\text{output} = \text{LocalAttn}(Q, K, V) + \sigma(\beta) \cdot \text{MemAttn}(Q, M)
\tag{1.1}
$$

The gate $\sigma(\beta)$ is initialized close to zero, so the model starts by 
reproducing standard local attention and gradually learns to use the memory.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $N_\ell$ attention layers, 
we seek a modified attention block $\mathcal{A}'$ such that:

1. $\mathcal{A}'$ reproduces the pretrained behavior at initialization.
2. $\mathcal{A}'$ reduces attention cost from $\mathcal{O}(L^2)$ to 
   $\mathcal{O}(L \cdot W + L \cdot d_c)$.
3. $\mathcal{A}'$ maintains access to global context via the memory matrix.
4. $\mathcal{A}'$ works across four attention architectures.

We do **not** aim to reduce model size. Infini-Attention targets 
**inference efficiency** for long sequences, not model compression.

## 1.3 Our Approach: Infini-Attention

Infini-Attention replaces the standard MHSA block with:

1. **Frozen base projections** $W_Q, W_K, W_V, W_O$ transferred from the 
   pretrained attention.
2. **Trainable LoRA adapters** on each projection.
3. **A learned compressive memory** $M$ per head.
4. **A learnable scalar gate** $\beta$.

The local branch uses sliding-window attention, which for causal models 
combines a causal mask and a distance-band mask:

$$
\text{LocalAttn}_{ij} = \text{softmax}\!\left(\frac{Q_i K_j^\top}{\sqrt{d_h}} + M_{ij}\right), \quad
M_{ij} = \begin{cases} 0 & j \leq i \text{ and } i - j < W \\ -\infty & \text{otherwise} \end{cases}
\tag{1.2}
$$

The memory branch computes:

$$
\text{MemAttn}_i = \text{softmax}\!\left(\frac{Q_i M^\top}{\sqrt{d_h}}\right) M
\tag{1.3}
$$

## 1.4 Contributions

- **A hybrid local+global attention block.** Infini-Attention combines 
  sliding-window local attention with a learned compressive memory via a 
  gated scalar.
- **Multi-architecture support.** Same codebase handles flat, nested, 
  causal, and GQA+RoPE models.
- **Initialization strategy.** The gate $\beta = -5$ ensures the memory 
  starts nearly silent, preserving the pretrained local behavior.
- **Systematic evaluation.** Benchmarks on DistilBERT, RoBERTa, and GPT-2 
  with ablations over $\beta$ initialization, compression dimension, and 
  window size.
- **Honest reporting.** Parameter count increases; the "compression" in the 
  name refers to the memory, not the model.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background. Section 4 
presents Infini-Attention. Section 5 details multi-architecture integration. 
Section 6 gives setup. Section 7 reports results. Section 8 analyzes 
contributions. Section 9 provides guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $W$ | Sliding window size |
| $H$ | Number of heads |
| $d_h$ | Per-head dimension |
| $d_c$ | Compression dimension |
| $M$ | Compressive memory matrix |
| $\beta$ | Gating scalar |
| $\sigma$ | Sigmoid |
| $W_Q, W_K, W_V, W_O$ | Projection matrices |
| $r$ | LoRA rank |

---