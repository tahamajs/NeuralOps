# 1. Introduction

## 1.1 Motivation

Attention dominates the compute and memory cost of transformer inference:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

The naive implementation materializes the $L \times L$ attention matrix, 
requiring $\mathcal{O}(L^2)$ memory. For $L = 8192$ and $d_h = 128$, this 
is 2 GB **per head, per layer** in FP16 — infeasible for most GPUs.

**FlashAttention-3** (Dao et al., 2024) solves this via **tiling** and 
**recomputation**: the attention matrix is computed in blocks that fit in 
GPU shared memory, and intermediate activations are recomputed during 
backward rather than stored. This gives:

- **Memory:** $\mathcal{O}(L)$ instead of $\mathcal{O}(L^2)$.
- **Throughput:** 2-4× faster than standard attention on Hopper GPUs.

**FP8 support.** FlashAttention-3 adds FP8 (e4m3) tensor core utilization, 
achieving another 2× on H100/H200.

## 1.2 The Deployment Problem

Despite its benefits, FlashAttention-3 is not a drop-in replacement:

1. **Hardware requirement.** Requires NVIDIA Hopper (H100/H200) for 
   TMA and FP8. Ampere (A100) supports FlashAttention-2 only; older GPUs 
   use the SDPA fallback.
2. **Software dependency.** Requires the `flash-attn` or `flash-attn-3` 
   Python package to be installed.
3. **Architecture heterogeneity.** Pretrained models differ in their 
   attention organization (fused Conv1D, nested self-attention, GQA, etc.), 
   requiring per-architecture weight extraction.

Our wrapper addresses all three:

- **Automatic fallback.** If FlashAttention-3 is not available, use 
  PyTorch's SDPA kernel (which itself may use FlashAttention-2 internally).
- **Multi-architecture weight copy.** A single `copy_weights` handles 
  GPT-2 Conv1D, RoBERTa/BERT nested, DistilBERT flat, and Qwen/LLaMA GQA.
- **FP8 opt-in.** FP8 precision is enabled via a flag, with automatic 
  dtype conversion and graceful NaN handling.

## 1.3 Problem Statement

Given a pretrained attention module $\mathcal{A}$ with projections 
$W_Q, W_K, W_V, W_O$, we seek a wrapper $\mathcal{A}'$ such that:

1. **Exactness.** $\mathcal{A}'(X) = \mathcal{A}(X)$ mathematically (up to 
   numerical precision).
2. **Efficiency.** When FlashAttention-3 is available, 
   $\text{Time}(\mathcal{A}') < \text{Time}(\mathcal{A})$ on Hopper GPUs.
3. **Robustness.** When FlashAttention-3 is not available, gracefully fall 
   back to SDPA without crashing.
4. **Portability.** Supports flat, nested, and causal architectures, plus 
   GQA and RoPE.
5. **Recovery capacity.** Add trainable LoRA adapters for task adaptation.

**Note.** This is **not** a compression method. Parameter count increases 
by the LoRA adapter size.

## 1.4 Our Approach

We provide a wrapper class `FlashAttention3` that:

1. **Extracts pretrained weights** from the source attention module.
2. **Freezes base weights** and adds trainable LoRA adapters.
3. **Dispatches** to FlashAttention-3 if available, otherwise to SDPA.
4. **Supports GQA** via `repeat_kv` before the attention call.
5. **Applies RoPE** (with partial-dim support) before the attention call.
6. **Handles causal masking** for decoder models.
7. **Supports cross-attention** for encoder-decoder models.

## 1.5 Contributions

- **A production-hardened FlashAttention-3 wrapper.** Works across four 
  architectures with exact weight transfer and graceful fallback.
- **Multi-architecture weight extraction.** Handles GPT-2 Conv1D, 
  RoBERTa/BERT nested, DistilBERT flat, and Qwen/LLaMA GQA.
- **FP8 support with automatic fallback.** FP8 precision on Hopper GPUs 
  with FP16 fallback if the hardware or library doesn't support it.
- **Honest reporting.** We disclose that on non-Hopper GPUs (like our 
  RTX 4070 test machine), the SDPA fallback is used, not the FlashAttention-3 
  kernel.
- **Full reproducibility.** Code and evaluation scripts released under 
  Apache 2.0.

## 1.6 Paper Organization

Section 2 reviews related work. Section 3 provides background on 
FlashAttention, SDPA, and hardware acceleration. Section 4 describes the 
wrapper. Section 5 details architecture integration. Section 6 gives 
experimental setup. Section 7 reports results. Section 8 analyzes the 
crossover between kernel and fallback paths. Section 9 discusses practical 
usage. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Per-head dimension |
| $H$ | Number of heads |
| $H_{kv}$ | Number of KV heads |
| $B$ | Block size in FlashAttention tiling |
| $\mathcal{A}$ | Original attention module |
| $\mathcal{A}'$ | Wrapper |
| $r_L$ | LoRA rank |

---