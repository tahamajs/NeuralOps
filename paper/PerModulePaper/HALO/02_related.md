<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Linear Attention

**Linear Transformer** (Katharopoulos et al., 2020) replaces softmax with 
$\phi(x) = \mathrm{elu}(x) + 1$, achieving $\mathcal{O}(L)$ complexity via 
the associativity of matrix multiplication.

**Performer** (Choromanski et al., 2021) uses positive orthogonal random 
features (PORF) for unbiased approximation of softmax.

**Liquid** (Anonymous et al., 2024) uses a learned kernel for linear attention.

**HALO-ReGLA's linear branch** uses the same $\mathrm{elu} + 1$ kernel, but 
in a **hybrid** setting with a **dense branch** and **learnable gates**.

## 2.2 Hybrid Attention

**H-Transformer** (Zhu & Soricut, 2021) alternates linear and dense layers.

**Hyena** (Poli et al., 2023) interleaves implicit convolutions with attention.

**NHA** (our framework) uses local windows + global RFF.

**HALO-ReGLA** differs by running the two streams **in parallel within the 
same layer** and combining them via **per-head learnable gates**.

## 2.3 Gated Attention

**FLASH** (Hua et al., 2022) uses gated attention with learned gates between 
linear and softmax branches. **HALO-ReGLA** is closely related but 
distinguishes itself by:
- Rectified kernel (positive features).
- Explicit prefix-sum formulation.
- Per-head gating with bias initialization.

**GLA** (Yang et al., 2023) uses gated linear attention with data-dependent 
decay, which is orthogonal to our fusion gate.

## 2.4 Pretrained Weight Transfer

**LoRA** (Hu et al., 2021) enables training adapters on frozen base weights.

**HALO-ReGLA** transfers Q, K, V, O projections from pretrained weights and 
trains only LoRA + gate parameters, preserving the pretrained knowledge.

## 2.5 Position of Our Work

| Axis | Prior | HALO-ReGLA |
|---|---|---|
| Architecture | Fixed or alternating | **Parallel hybrid** |
| Gate | Static or per-layer | **Per-head learnable** |
| Kernel | RFF or ELU | **Rectified ELU** |
| Pretrained transfer | Rarely | **Full support** |
| Architectures | Limited | **BERT, RoBERTa, GPT-2, Qwen** |

**Key distinction.** HALO-ReGLA combines:
1. Parallel dense + linear streams.
2. Per-head gates with principled bias initialization.
3. Rectified kernel for stability.
4. Full pretrained weight transfer.

## 2.6 Comparison Table

| Method | Parallel | Per-head Gate | Pretrained | RoPE | GQA |
|---|---|---|---|---|---|
| H-Transformer | No | No | No | No | No |
| FLASH | Yes | Layer | No | No | No |
| Hyena | No | No | No | No | No |
| **HALO-ReGLA** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

---
