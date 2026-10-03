# 1. Introduction

## 1.1 Motivation

Transformer self-attention scales quadratically with sequence length $L$, 
in both compute and memory. For input $X \in \mathbb{R}^{B \times L \times d}$ 
with $H$ heads, the intermediate attention matrix $Q K^\top \in \mathbb{R}^{L \times L}$ 
costs $\mathcal{O}(L^2)$ per batch item. At $L = 4096$, this is ~16M 
entries per head, becoming prohibitive for long-context workloads.

Existing methods address this along three axes:

1. **Sparse attention** (Longformer, BigBird) restricts the attention 
   pattern to $\mathcal{O}(L \sqrt{L})$ connections.
2. **Low-rank attention** (Linformer, Nyströmformer) projects $K, V$ to a 
   smaller space.
3. **Kernel attention** (Performer, Linear Transformer) factorizes the 
   softmax kernel into feature maps, achieving $\mathcal{O}(L)$.

Each has trade-offs. Sparse methods lose global context; low-rank methods 
require fixed rank; kernel methods suffer from approximation error on 
peaked distributions.

**Hybrid attention** combines multiple strategies. AEGIS-V5 combines local 
windowed attention (exact, preserves local detail) with global kernel 
linear attention (approximate, captures long-range dependencies).

## 1.2 Problem Statement

Given a pretrained transformer $\mathcal{M}$ with $H$ heads, we seek an 
attention mechanism $\mathcal{A}'$ such that:

$$
\mathcal{A}' \approx \mathcal{A}, \qquad
\text{Cost}(\mathcal{A}') \ll \mathcal{O}(L^2)
\tag{1.1}
$$

subject to:

1. **Multi-architecture compatibility:** Must work on DistilBERT (flat), 
   RoBERTa (nested), GPT-2 (Conv1D), and Qwen3.5 (GQA + RoPE).
2. **Causal correctness:** Decoders must preserve causality.
3. **Parameter efficiency:** Base weights frozen; only LoRA adapters trained.
4. **Numerical stability:** No NaNs, no silent collapses.

## 1.3 Our Approach

AEGIS-V5 has four key components:

### 1.3.1 Split-Head Architecture

We partition $H$ heads into:

- $H_\text{local} = \lceil 2H/3 \rceil$ heads using **local windowed attention**
- $H_\text{global} = H - H_\text{local}$ heads using **kernel linear attention**

This design lets the model allocate capacity adaptively: most heads focus 
on local structure, a few capture global context.

### 1.3.2 Energy-Based SVD Factorization

Each projection $W$ is decomposed via SVD, and the rank $r$ is chosen to 
capture a target **energy** fraction:

$$
r^* = \min \left\{ r : \frac{\sum_{i \leq r} \sigma_i^2}{\sum_i \sigma_i^2} \geq \eta \right\}
\tag{1.2}
$$

where $\eta = 0.95$ by default.

### 1.3.3 LoRA on Frozen SVD Factors

The SVD factors $(U_r, V_r)$ are frozen; LoRA adapters provide trainable 
capacity:

$$
\Delta W = \frac{\alpha}{r_L} B A
\tag{1.3}
$$

### 1.3.4 Per-Head Adaptive Gating

The local and global outputs are combined with a learnable per-head gate:

$$
\text{Out} = \sigma(g_h) \cdot \text{Out}_\text{local} + (1 - \sigma(g_h)) \cdot \text{Out}_\text{global}
\tag{1.4}
$$

Gate bias is initialized to $+2.0$, giving $\sigma(2.0) \approx 0.88$ 
(mostly local at start).

## 1.4 Contributions

We make the following contributions:

- **A hybrid split-head attention.** AEGIS-V5 combines local windowed and 
  global kernel attention, with per-head adaptive gating.
- **Energy-based SVD factorization.** Base projections are compressed with 
  a target energy fraction, giving explicit control over rank.
- **A production-hardened implementation.** Handles four architectures with 
  GQA, RoPE, nested encoders, Conv1D, and safe fallback.
- **Numerical stability fixes.** We replaced the original ReLU-based `softpick` 
  (which returned all-zeros on masked rows) with a stable softmax-based 
  variant, eliminating a subtle collapse mode.
- **Full reproducibility.** Code and checkpoints released.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on attention, 
SVD, LoRA, and kernel methods. Section 4 presents AEGIS-V5. Section 5 
describes integration. Section 6 details the setup. Section 7 reports 
results. Section 8 analyzes trade-offs. Section 9 discusses practical 
guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ or $N$ | Sequence length |
| $H$ | Total heads |
| $H_\text{local}$ | Heads using local attention |
| $H_\text{global}$ | Heads using global attention |
| $W$ | Local window size |
| $F$ | RFF feature dimension |
| $r$ | SVD rank |
| $\eta$ | Energy fraction threshold |
| $r_L$ | LoRA rank |
| $g_h$ | Per-head gate logit |
| $\sigma$ | Sigmoid |

---