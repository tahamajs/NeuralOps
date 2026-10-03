# 1. Introduction

## 1.1 Motivation

Transformer self-attention has quadratic complexity in sequence length, 
motivating a large body of work on efficient alternatives. **State-space 
models** (SSMs), particularly **Mamba** (Gu & Dao, 2023), have emerged as a 
compelling family: they achieve $\mathcal{O}(L)$ complexity via selective 
state updates with input-dependent dynamics.

However, replacing a pretrained transformer's attention with an SSM is 
non-trivial:

1. **Pretrained weights:** The SSM's input/output projections have no 
   correspondence to the pretrained $W_Q, W_K, W_V, W_O$.
2. **Residual flow:** SSMs process the input in a fundamentally different way, 
   changing the residual stream statistics.
3. **Training cost:** Training a full SSM from scratch is expensive.
4. **Causal vs bidirectional:** SSMs are naturally causal; encoders need 
   bidirectional processing.

**Mamba-Inspired Gated State-Space Attention** addresses these challenges:

- It **reuses pretrained weights** by projecting Q, K, V into a two-branch 
  gating structure.
- It applies a **learned exponential decay** to emulate selective state-space 
  dynamics without a true SSM scan.
- It adds a **preservation gate** that blends the new output with the residual 
  input, stabilizing the initial forward pass.
- It supports both causal (decoders) and bidirectional (encoders) modes.

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with attention projections 
$\{W_Q, W_K, W_V, W_O\}$, we seek a replacement mechanism $\mathcal{A}'$ such 
that:

$$
\mathcal{A}' \approx \mathcal{A}, \qquad
\mathcal{A}' \text{ has SSM-inspired structure}
\tag{1.1}
$$

subject to:

1. **Weight reuse:** Must copy pretrained weights via `copy_weights`.
2. **Multi-architecture:** DistilBERT, RoBERTa, GPT-2, Qwen.
3. **Causal and bidirectional:** Support both modes.
4. **Recovery capacity:** LoRA adapters for fine-tuning.
5. **Preservation:** Blend with residual input to stabilize.

## 1.3 Our Approach

We propose a gated state-space-inspired attention with:

### 1.3.1 Two-Branch In-Projection

The input projection is doubled: $W_\text{in} \in \mathbb{R}^{d \times 2d}$. 
The output splits into:

- $p_1$: SSM input representation (projected from $W_V$ or $W_Q$)
- $p_2$: gating representation (projected from $W_Q$ or $W_K$)

### 1.3.2 Learned Exponential Decay

For each dimension $i$, we learn a decay $\gamma_i \in (0, 1)$ via:

$$
\gamma_i = \sigma(-\exp(A_{\log, i}))
\tag{1.2}
$$

The pairwise weight between positions $j$ and $i$ ($j \leq i$):

$$
w_{ij}^{(d)} = \frac{\exp(\text{dist}_{ij} \cdot \log \gamma_d)}{\sum_{j'} \exp(\text{dist}_{ij'} \cdot \log \gamma_d)}
\tag{1.3}
$$

where $\text{dist}_{ij} = \max(0, j - i)$ for causal mode.

### 1.3.3 Gated Fusion

$$
\text{Out} = \text{SiLU}(p_1^\text{decayed}) \odot p_2
\tag{1.4}
$$

### 1.3.4 Preservation Gate

$$
\text{Final} = \sigma(g) \cdot \text{Out} + (1 - \sigma(g)) \cdot X
\tag{1.5}
$$

with $g$ initialized to 2.0 (so $\sigma(g) \approx 0.88$).

## 1.4 Contributions

- **A gated SSM-inspired attention.** Reuses pretrained Q/K/V/O weights in a 
  two-branch gated state-space formulation with learned exponential decay.
- **A preservation gate.** Blends the SSM output with the residual input, 
  stabilizing the initial forward pass.
- **Multi-architecture support.** Handles DistilBERT, RoBERTa, GPT-2, and 
  Qwen.
- **Causal and bidirectional modes.** Learned decay for encoders uses both 
  forward and backward directions.
- **Honest complexity reporting.** We acknowledge that the implementation is 
  $\mathcal{O}(L^2 D)$, **not** $\mathcal{O}(L)$, despite the SSM inspiration.

## 1.5 Paper Organization

Section 2 reviews related work on SSMs and efficient attention. Section 3 
provides background on state-space models, Mamba, and attention. Section 4 
presents our method. Section 5 describes integration. Section 6 details the 
setup. Section 7 reports results. Section 8 analyzes the SSM-inspired 
behavior. Section 9 discusses limitations and trade-offs. Section 10 lists 
limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d$ | Hidden dimension |
| $X \in \mathbb{R}^{B \times L \times d}$ | Input |
| $p_1, p_2$ | Two branches of in-projection |
| $A_{\log}$ | Log-decay parameter |
| $\gamma_i$ | Decay factor for dimension $i$ |
| $g$ | Preservation gate parameter |
| $r_L$ | LoRA rank |

---