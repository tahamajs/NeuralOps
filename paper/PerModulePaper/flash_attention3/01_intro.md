# 1. Introduction

## 1.1 Motivation

Standard scaled dot-product attention computes:

$$
\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_h}}\right) V
\tag{1.1}
$$

The intermediate matrix $S = QK^\top \in \mathbb{R}^{B \times H \times L \times L}$ 
requires $\mathcal{O}(L^2)$ memory. For $L = 8\text{K}$, $B = 1$, $H = 12$ 
at FP16, this is roughly **1.5 GB per layer** — the dominant memory cost 
of transformer inference.

**IO-aware attention** (FlashAttention, Dao et al., 2022) avoids 
materializing $S$ by **tiling** the computation:

- Load a tile of Q into SRAM.
- Compute softmax incrementally using running max and running sum.
- Never write the full $S$ matrix to HBM.

This gives $\mathcal{O}(L)$ **peak memory** and **2–4× wall-clock speedup** 
over standard attention.

**FlashAttention-3** (Shah et al., 2024) extends this to Hopper 
architecture:

1. **Asynchronous TMA** (Tensor Memory Accelerator) for bulk memory copy.
2. **Warp specialization** to overlap compute and memory.
3. **FP8 support** for $2\times$ higher throughput.

Crucially, FlashAttention-3 produces **exact** attention — the output is 
mathematically identical to standard attention.

## 1.2 Problem Statement

Given a pretrained transformer with standard softmax attention, we seek a 
drop-in integration of FlashAttention-3 such that:

1. **Exact attention.** No approximation error.
2. **Weight compatibility.** Pretrained weights transfer correctly.
3. **Graceful fallback.** When FA3 is unavailable, use PyTorch SDPA.
4. **Parameter efficiency.** LoRA adapters for recovery fine-tuning.
5. **Multi-architecture.** DistilBERT, RoBERTa, GPT-2, Qwen3.5.
6. **Cross-attention support.**

## 1.3 Our Approach

We integrate FlashAttention-3 with:

### 1.3.1 Dispatch Logic

```python
if FLASH_ATTN_AVAILABLE and hidden_states.is_cuda:
    try:
        out = flash_attn_func(q, k, v, causal=causal_flag)
    except Exception:
        out = None

if out is None:
    out = F.scaled_dot_product_attention(q, k, v, ...)
```

Two code paths:
1. **FA3** — the optimized kernel.
2. **SDPA** — PyTorch's built-in exact attention.

### 1.3.2 FP8 Mode

$$
Q_\text{fp8} = \text{cast}_{e4m3fn}(Q), \quad
K_\text{fp8} = \text{cast}_{e4m3fn}(K), \quad
V_\text{fp8} = \text{cast}_{e4m3fn}(V)
\tag{1.2}
$$

FA3 computes the attention in FP8, then casts the output back.

### 1.3.3 Weight Transfer

Base weights from the pretrained attention are copied into `LoRALinear` 
modules. The Q, K, V, O projections are transferred exactly; LoRA adapters 
provide fine-tuning capacity.

## 1.4 Contributions

- **A production-hardened integration.** We handle weight transfer across 
  four architectures, GQA, RoPE, causal masking, cross-attention, and 
  fallback paths.
- **FP8 support.** Optional FP8 casting for Hopper Tensor Cores.
- **Graceful fallback.** When FA3 is unavailable, SDPA provides 
  functionally-equivalent output.
- **Honest framing.** We state explicitly that this is **not a compression 
  method** — parameter count is unchanged (modulo LoRA).
- **Full reproducibility.** Code, weights, and scripts.

## 1.5 Paper Organization

Section 2 reviews related work. Section 3 provides background on 
FlashAttention, SDPA, and Hopper TMA. Section 4 presents our integration. 
Section 5 describes cross-architecture support. Section 6 details the setup. 
Section 7 reports results. Section 8 analyzes fallback and FP8 behavior. 
Section 9 discusses trade-offs. Section 10 lists limitations. Section 11 
concludes.

---

## Notation

| Symbol | Meaning |
|---|---|
| $L$ | Sequence length |
| $d_h$ | Head dimension |
| $H$ | Number of query heads |
| $H_{kv}$ | Number of KV heads |
| $B$ | Batch size |
| $\sigma$ | Softmax function |
| $\text{cast}_{e4m3fn}$ | FP8 E4M3 cast |
| SDPA | Scaled Dot-Product Attention |
| TMA | Tensor Memory Accelerator |
| HBM | High-Bandwidth Memory |
| SRAM | Shared memory on chip |

---
