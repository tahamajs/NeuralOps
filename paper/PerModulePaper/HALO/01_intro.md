<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

Transformer inference at long sequence length is dominated by the 
$\mathcal{O}(L^2)$ cost of softmax attention. Two broad strategies have 
emerged to reduce this cost:

1. **Dense softmax attention** $\mathcal{A}_\text{dense}(Q, K, V) = 
   \text{softmax}(QK^\top / \sqrt{d}) V$. This preserves the expressive power 
   of exact attention but requires materializing an $L \times L$ matrix.

2. **Linear attention** $\mathcal{A}_\text{lin}(Q, K, V) = 
   \phi(Q) (\phi(K)^\top V) / (\phi(Q) \cdot \sum_j \phi(K_j))$ for a kernel 
   $\phi$. This reduces complexity to $\mathcal{O}(L)$ but trades quality.

Neither extreme is universally better. On tasks requiring fine-grained 
reasoning (e.g., in-context learning), dense attention wins. On retrieval-like 
tasks with local structure, linear attention matches dense at a fraction of 
the cost.

**The opportunity.** Rather than commit to one strategy, **hybrid attention** 
runs both and combines them. If the combination weight is learned per-head 
and per-token, the model can allocate dense capacity where it matters and 
linear capacity elsewhere.

## 1.2 Problem Statement

Given a pretrained transformer with dense attention, we seek a modified 
module $\mathcal{A}'$ such that:

1. **Efficiency:** $\mathcal{O}(L)$ memory and time for the linear branch, 
   at most $\mathcal{O}(L^2)$ for the dense branch when needed.
2. **Quality:** $\Delta\text{Acc} \leq 5\%$ on SST-2.
3. **Pretrained compatibility:** No retraining of base weights; only LoRA 
   and gating parameters are trainable.
4. **Architecture portability:** Flat, nested, and causal transformers.

## 1.3 Our Approach: HALO-ReGLA

We propose **HALO-ReGLA** (Hardware-Aware Linear Optics & Rectified Gated 
Linear Attention), which:

1. **Runs two attention streams in parallel.**
   - Dense: standard softmax attention.
   - Linear: rectified kernel attention using $\phi(x) = \mathrm{elu}(x) + 1$.

2. **Combines them via per-head learnable gates.** The gate for head $h$ at 
   position $i$ is $g_{h,i} = \sigma(w_h^\top x_i + b_h)$, and the output is 
   $g \odot O_\text{dense} + (1 - g) \odot O_\text{lin}$.

3. **Initializes the gate to favor dense behavior.** We set $b_h = 1.8$ so 
   $\sigma(1.8) \approx 0.86$ at initialization. This ensures the model 
   starts near the pretrained dense behavior.

4. **Uses a rectified kernel for numerical stability.** The non-negativity of 
   $\phi(x) = \mathrm{elu}(x) + 1$ guarantees the denominator 
   $\phi(Q) \cdot \sum_j \phi(K_j)$ is positive, avoiding division by zero.

5. **Preserves causality via prefix sums.** For causal models, the linear 
   branch uses a cumulative-sum formulation:
   $$
   O_i^\text{lin} = \frac{\phi(Q_i) \sum_{j \le i} \phi(K_j) V_j^\top}{\phi(Q_i) \sum_{j \le i} \phi(K_j)}
   $$

## 1.4 Contributions

- **A hybrid attention module.** HALO-ReGLA runs dense and linear streams 
  in parallel and combines them with per-head learnable gates.
- **A numerically stable formulation.** The rectified kernel 
  $\phi(x) = \mathrm{elu}(x) + 1$ ensures positive denominators in both 
  causal and bidirectional settings.
- **A production-hardened implementation.** Supports four architectures 
  with RoPE, GQA, LoRA recovery, and safe injection.
- **A systematic evaluation.** Benchmarked on DistilBERT, RoBERTa, and GPT-2 
  on SST-2 with ablations over gate bias, kernel choice, and fusion strategy.
- **Gate analysis.** We show that learned gates are non-uniform and 
  systematically depend on layer depth and head index.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on dense and 
linear attention. Section 4 presents HALO-ReGLA. Section 5 describes 
integration with four architectures. Section 6 details the experimental 
setup. Section 7 reports results. Section 8 analyzes the learned gates. 
Section 9 discusses practical guidance. Section 10 lists limitations. 
Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $H$ | Number of attention heads |
| $H_{kv}$ | Number of KV heads |
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $\phi(\cdot)$ | Feature map (rectified: $\mathrm{elu} + 1$) |
| $g$ | Learnable gate |
| $w_h, b_h$ | Gate parameters for head $h$ |
| $\mathcal{A}_\text{dense}$ | Dense softmax attention |
| $\mathcal{A}_\text{lin}$ | Linear attention |
| $\alpha$ | LoRA scaling |
| $r$ | LoRA rank |

---
