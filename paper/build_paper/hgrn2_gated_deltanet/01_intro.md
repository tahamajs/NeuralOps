# 1. Introduction

## 1.1 Motivation

The quadratic complexity of transformer attention, $\mathcal{O}(L^2)$ in 
sequence length $L$, has motivated a decade of research into efficient 
alternatives. Two broad families have emerged:

1. **Sparse attention** (Longformer, BigBird, Routing Transformer): restrict 
   the attention pattern to a subset of token pairs.
2. **Linear recurrent attention** (Linear Transformer, Performer, HGRN, Mamba, 
   DeltaNet): replace softmax attention with a recurrence that processes 
   tokens sequentially with a fixed-size state.

The second family has seen renewed interest because of the **linear** — 
$\mathcal{O}(L)$ — scaling in both time and memory. A recurring theme in this 
literature is that **no single recurrence mechanism dominates**: some excel 
at fast decay, others at long-range retention, others at content-based 
retrieval.

We ask: **can we combine two complementary recurrences in a single module 
with a learned fusion?**

## 1.2 Problem Statement

Given a transformer with a standard multi-head self-attention module 
$\mathcal{A}$ and a target of efficient long-context inference, we seek a 
replacement module $\mathcal{A}'$ such that:

1. **Linear complexity:** $\mathcal{A}'$ runs in $\mathcal{O}(L)$ time and memory.
2. **Expressivity:** $\mathcal{A}'$ can represent both fast-decay and 
   long-range dependencies.
3. **Pretrained compatibility:** Base weights are transferable from $\mathcal{A}$.
4. **Learnable recovery:** Only LoRA adapters and small modules are trained.

We do **not** require parameter reduction — attention-replacement methods 
are useful even when parameter count is roughly preserved.

## 1.3 Our Approach

**HGRN2-GatedDeltaNet** combines two linear recurrent branches:

### 1.3.1 HGRN-2 Branch

HGRN-2 (Qin et al., 2024) is an element-wise recurrence:

$$
s_t = \alpha_t \odot s_{t-1} + (1 - \alpha_t) \odot u_t
\tag{1.1}
$$

where $\alpha_t \in (0, 1)$ is a **bounded forget gate** and $u_t$ is a 
candidate update. HGRN-2 is efficient and stable, but its state is 
**diagonal** — it cannot represent cross-feature interactions.

### 1.3.2 Gated DeltaNet Branch

Gated DeltaNet maintains a **matrix state** $S_t \in \mathbb{R}^{d \times d}$:

$$
S_t = S_{t-1} + \beta_t (v_t - S_{t-1} k_t) k_t^\top
\tag{1.2}
$$

where the **Delta rule** $(v_t - S_{t-1} k_t)$ computes a prediction error. 
The matrix state captures cross-feature interactions, but requires more 
compute per step.

### 1.3.3 Learned Fusion

The two branches are combined via a **token-level gate**:

$$
Y = \gamma \odot Y_\text{HGRN} + (1 - \gamma) \odot Y_\text{DeltaNet}, \quad
\gamma = \sigma(W_g X)
\tag{1.3}
$$

The gate is initialized to $0.5$ (equal blend) and learns to prefer the 
branch that fits each token.

## 1.4 Contributions

- **A hybrid linear recurrent module.** We propose HGRN2-GatedDeltaNet, 
  which combines element-wise and matrix recurrences with a learned gate.
- **Robust weight transfer.** Our `copy_weights` handles flat, nested, and 
  causal attention architectures, with explicit failure when no projection 
  can be found.
- **An honest evaluation.** We report both the promising long-context potential 
  and the current **short-context slowdown**.
- **A production-hardened implementation.** Our PyTorch implementation 
  includes gate-bias initialization, per-branch failure handling, and 
  double-injection guards.
- **Full reproducibility.** Code, weights, and evaluation scripts are 
  released under Apache 2.0.

## 1.5 Paper Organization

Section 2 reviews related work in linear attention and recurrence. Section 3 
provides background on HGRN-2, DeltaNet, and gating. Section 4 presents 
HGRN2-GatedDeltaNet. Section 5 describes integration with three architectures. 
Section 6 details the setup. Section 7 reports results. Section 8 analyzes 
the trade-offs. Section 9 discusses practical guidance. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $H$ | Number of heads |
| $s_t$ | HGRN-2 element-wise state |
| $S_t$ | DeltaNet matrix state |
| $\alpha_t$ | Forget gate |
| $\beta_t$ | Delta update gate |
| $\gamma$ | Fusion gate |
| $\sigma$ | Sigmoid function |
| $\odot$ | Element-wise product |

---