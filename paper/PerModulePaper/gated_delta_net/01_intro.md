<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The quadratic cost of transformer self-attention — $\mathcal{O}(L^2)$ in 
both time and memory — has motivated a decade of research into efficient 
alternatives. Among these, **linear recurrent attention** is particularly 
attractive because it reduces complexity to $\mathcal{O}(L)$ while 
maintaining the same asymptotic expressiveness.

The core idea: replace the softmax attention matrix with a **fixed-size 
state** $S \in \mathbb{R}^{d_h \times d_h}$ that is updated as tokens arrive:

$$
S_t = f(S_{t-1}, K_t, V_t), \quad y_t = Q_t^\top S_t
\tag{1.1}
$$

For this to work, $f$ must be **associative** — that is, the composition 
$(f \circ f)$ must be efficiently computable.

0.85

<!-- 
EXAMPLE continuation:
"Early linear attention (Katharopoulos et al., 2020) used f(S, k, v) = S + k v^T, 
which is trivially associative but accumulates noise over long sequences. 
More recent work has introduced gating (RetNet, Mamba) and error correction 
(DeltaNet) to improve the state update."
-->

## 1.2 Problem Statement

We seek an attention mechanism $\mathcal{A}$ with the following properties:

1. **Linear complexity:** $\mathcal{O}(L)$ time and memory.
2. **Bidirectional support:** work on encoders (BERT, RoBERTa).
3. **Causal support:** work on decoders (GPT-2, LLaMA).
4. **Weight compatibility:** transfer pretrained Q/K/V/O projections.
5. **Gating:** input-dependent forgetting/retention.
6. **Fine-tunable:** LoRA recovery.

## 1.3 Our Approach

**Gated DeltaNet** combines three ideas:

1. **Delta Rule update** (Widrow–Hoff, 1960): correct the state by the error 
   between the predicted value and the observed value:
   $$
   S_t = S_{t-1} + K_t^\top (V_t - K_t S_{t-1})
   \tag{1.2}
   $$

2. **Input-dependent gating:** replace fixed step size with learnable 
   $\alpha_t, \beta_t \in (0, 1)$:
   $$
   S_t = \beta_t S_{t-1} + \alpha_t K_t^\top (V_t - K_t S_{t-1})
   \tag{1.3}
   $$

3. **Bidirectional scan for encoders:** average forward and backward scans:
   $$
   y = \frac{1}{2}(y_{\text{fwd}} + y_{\text{bwd}})
   \tag{1.4}
   $$

## 1.4 Contributions

- **A gated delta rule attention** with input-dependent decay and 
  bidirectional support.
- **A chunked implementation** that is exact (not approximate) but uses 
  nested Python loops.
- **An honest evaluation** reporting that the implementation is 
  **~45× slower** than baseline on DistilBERT and exceeds memory on larger 
  models.
- **A negative result:** the algorithm is not the bottleneck; the 
  implementation is.
- **Full reproducibility**, including code, timing measurements, and 
  recommendations for fixing.

## 1.5 Paper Organization

Section 2 reviews linear attention, state-space models, and DeltaNet. 
Section 3 provides background on RNNs, associative scans, and the Delta 
Rule. Section 4 presents Gated DeltaNet. Section 5 covers architecture 
integration. Section 6 details the experimental setup. Section 7 reports 
results (including timeouts). Section 8 analyzes the implementation 
bottleneck. Section 9 discusses practical guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §1.1 |
| $d$ | Head dimension | §1.1 |
| $H$ | Number of heads | §1.1 |
| $S_t$ | State matrix at time $t$ | §1.1 |
| $\alpha_t$ | Forget gate (input-dependent) | §1.3 |
| $\beta_t$ | Write gate (input-dependent) | §1.3 |
| $K_t, V_t, Q_t$ | Key, Value, Query at token $t$ | §1.1 |
| $r$ | LoRA rank | §3.5 |
| $\text{chunk\_size}$ | Inner loop size | §4.4 |
</parameter>

---