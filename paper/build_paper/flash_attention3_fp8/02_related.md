<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 02_related.md
     PURPOSE: Related work
     DEPENDENCIES: 01_intro.md
     LENGTH TARGET: 1 - 1.5 pages
     ═══════════════════════════════════════════════════════════════════════════ -->

# 2. Related Work

## 2.1 Efficient Exact Attention

### 2.1.1 FlashAttention Series

**FlashAttention** (Dao et al., 2022). Introduces tiling to keep the 
$L \times L$ attention matrix in SRAM. Memory drops from $\mathcal{O}(L^2)$ 
to $\mathcal{O}(L)$; recomputation in the backward pass avoids storing 
intermediate activations.

**FlashAttention-2** (Dao, 2023). Improves parallelism over the sequence 
dimension, reduces non-matmul FLOPs (rescaling, masking), and enables 
sequence-parallel training.

**FlashAttention-3** (Shah et al., 2024). Adds Hopper-specific features:
- **TMA (Tensor Memory Accelerator)** for asynchronous loads
- **Warpgroup specialization** (producer/consumer)
- **FP8 support** (e4m3 for Q/K, e5m2 for V or vice versa)
- **Pipelined softmax** overlapping with matrix multiplication

### 2.1.2 Memory-Efficient Attention

PyTorch's `F.scaled_dot_product_attention` (SDPA) dispatches to one of 
several backends: FlashAttention-2, memory-efficient attention (xFormers), 
or the native CUDA path. We use SDPA as the fallback.

### 2.1.3 Other Exact Attention

**Memory-efficient attention** (Rabe & Staats, 2021). Reduces memory by 
recomputing activations during backward.

**PagedAttention** (Kwon et al., 2023). Manages KV cache in pages for 
serving.

## 2.2 Low-Precision Attention

### 2.2.1 FP8 Training

**FP8 for Deep Learning** (Micikevicius et al., 2022). Defines the e4m3 and 
e5m2 formats and provides recipes for stable training.

**TransformerEngine** (NVIDIA, 2023). A library that automatically casts 
activations to FP8 during training.

**FP8-LM** (Peng et al., 2023). Shows that FP8 training matches BF16 
accuracy for large language models.

### 2.2.2 Quantization

**LLM.int8()** (Dettmers et al., 2022). 8-bit inference with outlier handling.

**GPTQ** (Frantar et al., 2023). Post-training quantization using second-order 
information.

**AWQ** (Lin et al., 2023). Activation-aware weight quantization.

## 2.3 Position of Our Work

Our implementation differs from prior work in three ways:

| Axis | Prior work | Ours |
|---|---|---|
| **Hardware assumption** | Assumes specific architecture | Detects at runtime |
| **Fallback path** | Error or requires reinstall | SDPA |
| **Architecture support** | Usually one family | Four families |
| **Weight transfer** | Not addressed | Exact copy |

**Key distinction.** The original FlashAttention-3 paper (Shah et al., 2024) 
reports speedups on H100. Our contribution is **portability** — a version 
that runs correctly on any GPU, using FP8 when available. We do not claim 
to match the H100 speedups on non-Hopper hardware.

---

## Comparison Table

| Implementation | FP8? | Fallback? | DistilBERT? | RoBERTa? | GPT-2? | Qwen? |
|---|---|---|---|---|---|---|
| FlashAttention-2 | No | Error | Manual | Manual | Manual | Manual |
| FlashAttention-3 | Yes | Error | Manual | Manual | Manual | Manual |
| PyTorch SDPA | No | — | Manual | Manual | Manual | Manual |
| **Ours** | **Yes** | **SDPA** | **Yes** | **Yes** | **Yes** | **Yes** |

---