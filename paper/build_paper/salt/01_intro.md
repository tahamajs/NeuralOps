<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Parameter-efficient fine-tuning (PEFT) has become the default approach for 
adapting large language models. Among PEFT methods, LoRA (Hu et al., 2021) 
is the most widely used: it freezes the base weight $W$ and adds a trainable 
rank-$r$ update:

$$
W_\text{LoRA} = W_\text{frozen} + \frac{\alpha}{r} B A
\tag{1.1}
$$

**Limitation.** LoRA reduces **trainable** parameters, but not **total** 
parameters. The base weight $W_\text{frozen}$ remains at full size. For a 
7B-parameter model, LoRA reduces the trainable count from 7B to ~10M but 
the checkpoint still weighs 7B parameters.

**Alternative: low-rank weights.** If $W$ itself is low-rank, we can store 
it as $W = U V$ with $r \ll \min(d_\text{in}, d_\text{out})$. This reduces 
total parameters. But it also throws away information: the base model no 
longer matches its pretrained state, and downstream accuracy may collapse.

0.85

<!-- 
EXAMPLE continuation:
"SALT bridges these two approaches. It stores the top-r singular vectors and 
values of W as fixed buffers, then adds a trainable per-singular-value scale 
γ and a low-rank residual AB. The result is a checkpoint that is 12× smaller 
than the full W, that starts as a rank-r SVD of the pretrained weights, and 
that can be adapted with a small number of trainable parameters."
-->

## 1.2 Problem Statement

We seek a parameterization $\mathcal{P}(W)$ such that:

1. **Total parameter reduction:** $|\mathcal{P}(W)| \ll |W|$
2. **Expressiveness:** $\mathcal{P}$ can represent a rich family of matrices, 
   not just rank-$r$ ones
3. **Pretrained initialization:** $\mathcal{P}$ can be initialized to closely 
   match a given $W_\text{pretrained}$
4. **Trainability:** a small subset of $\mathcal{P}$ is trainable

## 1.3 Our Approach

SALT parameterizes $W$ as:

$$
W = U_{:,:r} \cdot \text{diag}(\Sigma_{:r} \odot \gamma) \cdot V_{:r,:}^\top + A B
\tag{1.2}
$$

where:

- $U_{:,:r} \in \mathbb{R}^{d_\text{out} \times r}$, 
  $\Sigma_{:r} \in \mathbb{R}^r$, 
  $V_{:r,:} \in \mathbb{R}^{r \times d_\text{in}}$ — **buffers** from the SVD of $W_\text{pretrained}$
- $\gamma \in \mathbb{R}^r$ — trainable per-singular-value scale (init: 1)
- $A \in \mathbb{R}^{d_\text{out} \times r}$, 
  $B \in \mathbb{R}^{r \times d_\text{in}}$ — trainable low-rank residual (init: A small random, B = 0)

**At initialization:**
- $\gamma = \mathbf{1} \Rightarrow U \text{diag}(\Sigma) V^\top$ = truncated SVD of $W_\text{pretrained}$
- $A B = 0 \Rightarrow$ no residual contribution

So SALT starts as a rank-$r$ approximation of the pretrained weight.

## 1.4 Contributions

- **A new parameterization** that combines truncated SVD with trainable 
  singular-value modulation and a low-rank residual.
- **A 12× parameter reduction** for 768×768 matrices at rank 16.
- **A complete implementation** supporting four transformer families with 
  multi-architecture weight extraction.
- **An honest evaluation** showing that SALT preserves accuracy within 
  **0.85%** but that our naive forward implementation does not 
  realize inference-time benefits.

## 1.5 Paper Organization

Section 2 reviews low-rank adaptation and factorization methods. Section 3 
provides background on SVD, LoRA, and rank-$r$ approximation. Section 4 
presents SALT. Section 5 describes integration. Section 6 details the setup. 
Section 7 reports results. Section 8 analyzes the trade-off between storage 
and compute. Section 9 discusses guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $W$ | Weight matrix | §1.1 |
| $d_\text{in}, d_\text{out}$ | Input/output dims | §1.1 |
| $r$ | SALT rank | §1.1 |
| $U, \Sigma, V$ | SVD factors | §3.1 |
| $\gamma$ | Singular value scale | §1.3 |
| $A, B$ | Low-rank residual | §1.3 |
| $\alpha$ | LoRA scaling | §3.4 |
</parameter>

---