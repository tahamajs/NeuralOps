# 2. Related Work

## 2.1 Attention Kernel Optimization

**FlashAttention (v1).** Dao et al. (2022) introduce tiling and 
recomputation for attention, reducing memory from $\mathcal{O}(L^2)$ to 
$\mathcal{O}(L)$ while being faster due to reduced HBM traffic.

**FlashAttention-2.** Dao (2023) improves parallelism and work partitioning, 
achieving ~2× over v1.

**FlashAttention-3.** Dao et al. (2024) add TMA (Hopper), warp-specialized 
asynchronous execution, and FP8 support, achieving ~2× over v2 on H100.

**PyTorch SDPA.** PyTorch 2.0+ includes `F.scaled_dot_product_attention`, 
which auto-selects between a math path, a memory-efficient path, and 
FlashAttention-2 (if available). It's our fallback.

**xFormers.** Facebook's memory-efficient attention. Similar goals; 
superseded by SDPA.

## 2.2 Exact vs Approximate Attention

| Method | Exact? | Complexity | Memory |
|---|---|---|---|
| Naive | Yes | $O(L^2 d)$ | $O(L^2)$ |
| FlashAttention | Yes | $O(L^2 d)$ | $O(L)$ |
| SDPA (memory-efficient) | Yes | $O(L^2 d)$ | $O(L)$ |
| Performer | No | $O(L m d)$ | $O(L m)$ |
| Longformer | No (sparse) | $O(L W d)$ | $O(L W)$ |
| ClusterFormer | Approx. | $O(L K d)$ | $O(L K)$ |

FlashAttention is **exact** — the output is mathematically identical to 
the naive implementation. Only the memory layout and compute order change.

## 2.3 Hardware-Aware Transformer Optimization

**vLLM.** Kwon et al. (2023) provide PagedAttention for efficient 
KV-cache management. Complementary to FlashAttention.

**TensorRT.** NVIDIA's inference optimizer. Uses many of the same 
techniques as FlashAttention.

**DeepSpeed.** Microsoft's training framework. Includes memory-efficient 
attention.

## 2.4 Position of Our Work

| Feature | Raw FlashAttention | Our Wrapper |
|---|---|---|
| Install effort | Manual | Automatic |
| Fallback | None | SDPA |
| Architecture support | Manual | Multi-arch |
| Weight transfer | Manual | Automatic |
| LoRA adapters | Manual | Built-in |
| GQA, RoPE | Partial | Full |

Our wrapper is **not** a novel attention algorithm. It's a **production 
integration layer** that makes FlashAttention-3 usable in a research 
framework with pretrained models.

## 2.5 Comparison Table

| Method | Type | Complexity | Hardware | Exact |
|---|---|---|---|---|
| Naive | Baseline | $O(L^2)$ | Any | Yes |
| SDPA | Kernel | $O(L^2)$ memory $O(L)$ | Any (CUDA) | Yes |
| FlashAttention-2 | Kernel | $O(L^2)$ memory $O(L)$ | Ampere+ | Yes |
| **FlashAttention-3** | Kernel | $O(L^2)$ memory $O(L)$ | Hopper | Yes |
| Performer | Approx. | $O(L)$ | Any | No |

---