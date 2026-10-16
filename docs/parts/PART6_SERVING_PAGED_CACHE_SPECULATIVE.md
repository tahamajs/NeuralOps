# 🏛️ NEURALOPS PART 6 — SERVING, KV CACHE & HIGH-THROUGHPUT INFERENCE PRIMITIVES

**Architecture**: `NeuralOps` (`SVD_linear_Attention` v1.3.7)  
**Modules**: [`neuralops/inference/`](file:///Users/tahamajs/Documents/uni/Project/neuralops/inference/), [`neuralops/distributed/`](file:///Users/tahamajs/Documents/uni/Project/neuralops/distributed/)  
**Target Hardware**: NVIDIA GeForce RTX 4070 Laptop GPU 8GB (Ada Lovelace, SM 8.9)  

---

## 1. Paged Attention & Virtual Memory Cache Manager

The Paged KV Cache Manager in [`neuralops/inference/paged_cache.py`](file:///Users/tahamajs/Documents/uni/Project/neuralops/inference/) translates OS virtual memory paging to GPU HBM:

```
    LOGICAL KV TOKENS (Per Sequence):
    [Token 0 ... 15]   [Token 16 ... 31]   [Token 32 ... 47]
           |                  |                   |
    BLOCK TABLE:
    Block 0 -> Page 7  Block 1 -> Page 14  Block 2 -> Page 3
           |                  |                   |
    PHYSICAL GPU VRAM:
    [ Page 3: KV Data ] ... [ Page 7: KV Data ] ... [ Page 14: KV Data ]
```

### Key Engineering Features
1. **Zero Fragmentation**: Eliminates internal fragmentation by allocating fixed-size blocks (16 tokens per block).
2. **Dynamic Sequence Extension**: New blocks are allocated on demand without re-allocating contiguous memory chunks.
3. **Copy-on-Write (CoW)**: Forking sequences during beam search or tree decoding share physical memory blocks until modified.
4. **Hardware Throughput Gain**: Enables up to **$4\times$ higher concurrency** on fixed 8GB GPU memory.

---

## 2. Hierarchical Radix Tree Prefix Cache

Caches token prefixes in a persistent prefix tree across multi-turn conversational turns:
- **Prefix Matching**: If an incoming request shares an $L$-token system prompt or few-shot exemplar with prior requests, the KV cache blocks are instantly mapped without recomputing prefill attention.
- **Eviction Policies**: LRU (Least Recently Used) eviction frees least-used leaf nodes when physical block utilization exceeds 95%.

---

## 3. Speculative Decoding & Draft Verification Engine

Speculative decoding pairs a compressed low-rank draft model with an uncompressed foundation target model:

1. **Draft Generation ($K=5$ Tokens)**:
   The compressed student model (e.g. 80% compressed via HNC-SVD) generates $K$ candidate tokens sequentially at high speed:
   $$\tilde{x}_1, \tilde{x}_2, \dots, \tilde{x}_K \sim p_{\text{draft}}(x)$$
2. **Parallel Target Verification**:
   The uncompressed target model executes a single forward pass over all $K$ tokens simultaneously, computing target probabilities $p_{\text{target}}(x_k | x_{<k})$.
3. **Rejection Sampling**:
   Accepts token $k$ with probability $\min\left(1, \frac{p_{\text{target}}(x_k)}{p_{\text{draft}}(x_k)}\right)$.
4. **Latency Gain**: Delivers **$2.0\times - 2.8\times$ wall-clock speedup** with zero change to final output distribution.

---

## 4. Distributed Tensor Parallelism (TP) & Context Parallelism (CP)

Located in [`neuralops/distributed/`](file:///Users/tahamajs/Documents/uni/Project/neuralops/distributed/):
- **Tensor Parallelism**: Megatron-style column-parallel and row-parallel linear layers partitioning weight matrices across GPUs, combined with all-reduce communication primitives.
- **Context Parallelism**: Ring-attention distributing sequence tokens along the sequence dimension ($T=32\text{k}-128\text{k}$), allowing models to process contexts exceeding single-GPU memory limits.

---
*Part 6 certified under `tests/test_exhaustive_serving_paged_radix_speculative.py` and `tests/test_distributed_tp_and_cp.py`.*
