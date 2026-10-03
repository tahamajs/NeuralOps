<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 01_intro.md
     PURPOSE: Introduction
     DEPENDENCIES: 00_meta.md
     LENGTH TARGET: 1.5 - 2 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 1. Introduction

## 1.1 Motivation

The memory cost of attention has been a bottleneck for transformer inference 
since Vaswani et al. (2017). The intermediate attention matrix $Q K^\top \in 
\mathbb{R}^{L \times L}$ requires $\mathcal{O}(L^2)$ memory, which for 
$L = 8192$ and 16-bit precision is 128 MB per head — infeasible for 
long-context applications.

FlashAttention (Dao et al., 2022) solved this by **tiling** the attention 
computation in GPU SRAM, achieving exact attention with $\mathcal{O}(L)$ 
memory. FlashAttention-2 (Dao, 2023) improved parallelism across the sequence 
dimension and reduced non-matmul FLOPs. FlashAttention-3 (Shah et al., 2024) 
added **asynchronous hardware overlap** on Hopper GPUs and, crucially, 
**FP8 tensor core support**, which roughly doubles throughput on H100-class 
hardware.

0.85

<!-- 
EXAMPLE continuation:
"However, FP8 tensor cores are a hardware feature, not a software one. They 
exist on NVIDIA Hopper (H100, H200) and Ada (RTX 4090, L40) GPUs but are 
absent on Ampere (A100, RTX 30-series) and consumer Turing (RTX 20-series). 
A practitioner deploying on a mixture of hardware needs a code path that 
exploits FP8 when available and falls back gracefully when not."
-->

## 1.2 Problem Statement

We seek an attention implementation $\mathcal{A}$ such that:

1. **Exact or near-exact**: $\mathcal{A}(Q, K, V) \approx \text{softmax}(QK^\top/\sqrt{d})V$
2. **Memory-efficient**: $\mathcal{O}(L)$ peak memory, not $\mathcal{O}(L^2)$
3. **Hardware-portable**: works on any CUDA GPU, using FP8 when available
4. **Weight-compatible**: preserves pretrained weights without retraining
5. **Fine-tunable**: supports LoRA recovery when needed

## 1.3 Our Approach

We implement FlashAttention-3 with three design principles:

1. **Runtime hardware detection.** At construction time, we check for 
   `torch.float8_e4m3fn` support and the presence of `flash_attn` / 
   `flash_attn_interface`. The FP8 path is used only if both are available.

2. **Correct fallback to SDPA.** If the FP8 path is unavailable, we use 
   PyTorch's built-in `F.scaled_dot_product_attention`, which internally 
   dispatches to FlashAttention-2 or memory-efficient attention depending on 
   the CUDA version.

3. **Exact weight transfer.** Base weights are copied from the pretrained 
   attention with architecture-specific handling for Conv1D (GPT-2), nested 
   self-attention (RoBERTa), and GQA (Qwen).

## 1.4 Contributions

- **A hardware-aware FP8 attention implementation** that runs correctly 
  across four transformer families and gracefully degrades on non-Hopper 
  hardware.
- **An honest evaluation** reporting that on our test hardware (RTX 4070 
  Laptop), FP8 tensor cores are not exposed, so the FP8 path is never 
  taken. We identify the exact hardware needed to reproduce the FP8 
  speedups.
- **An accuracy-preservation study** showing that when the fallback path is 
  used, accuracy matches the baseline within **-0.92%** on DistilBERT, 
  **0.85%** on RoBERTa, and **0.85%** on GPT-2.
- **Full reproducibility**, including hardware-detection logic and LoRA 
  configuration.

## 1.5 Paper Organization

Section 2 reviews related work on FlashAttention and FP8 training. Section 3 
provides background on attention, FP8 formats, and LoRA. Section 4 presents 
our implementation. Section 5 describes architecture integration. Section 6 
details the experimental setup. Section 7 reports results. Section 8 
analyzes why FP8 is unavailable on our hardware. Section 9 discusses 
practical guidance. Section 10 lists limitations. Section 11 concludes.

---

## Notation

| Symbol | Meaning | First appears |
|---|---|---|
| $L$ | Sequence length | §3.1 |
| $d$ | Head dimension | §3.1 |
| $H$ | Number of query heads | §3.1 |
| $H_{kv}$ | Number of KV heads | §3.2 |
| $B$ | Batch size | §3.1 |
| $S$ | SRAM tile size (FlashAttention) | §3.3 |
| $r$ | LoRA rank | §3.5 |
| $\text{fp8}$ | 8-bit float format (e4m3fn or e5m2) | §3.4 |
</parameter>

---