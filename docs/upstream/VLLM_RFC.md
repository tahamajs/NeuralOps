# [RFC] Integration of NeuralOps Paged Recurrent State Cache in vLLM

- **RFC Target**: `vllm-project/vllm`
- **Author**: Taha Majlesi (`neuralops` maintainers)
- **Status**: Proposed / Upstream RFC
- **Date**: October 2026

---

## 1. Motivation

vLLM's PagedAttention revolutionized transformer inference by partitioning standard multi-head attention KV caches into fixed-size virtual blocks. However, emerging subquadratic architectures (e.g. Gated DeltaNet, GLA, RetNet, RWKV, Mamba, and SVD Linear Attention) maintain fixed-dimensional recurrent state tensors $S_t \in \mathbb{R}^{B \times H \times D \times D}$ rather than token-accumulating $O(T)$ KV matrices.

When deploying hybrid or pure subquadratic models in vLLM, allocating contiguous $O(T)$ buffers wastes GPU VRAM and violates vLLM's memory management paradigm.

---

## 2. Proposed Architecture: `PagedStateTable`

We propose integrating `PagedStateTable` (`neuralops.models.paged_state.PagedStateTable`) as an official vLLM state management backend for linear recurrent models:

```
+-------------------------------------------------------------------+
|                        vLLM BlockManager                          |
+-------------------------------------------------------------------+
                                  |
         +------------------------+------------------------+
         |                                                 |
         v                                                 v
Standard Attention                                Linear Recurrence
(K, V) Tokens Paging                           Fixed Block PagedStateTable
[Block Size: 16 tokens]                        [Block: H x D x D state]
```

### Key Capabilities
1. **Virtual Page Allocation**: Maps dynamic sequence IDs to non-contiguous recurrent state blocks in physical GPU memory.
2. **Copy-on-Write (CoW)**: Enables $O(1)$ zero-copy sequence forking for parallel sampling and beam search. When a sequence branches, child sequences share the parent page until the first autoregressive recurrent step.
3. **Constant VRAM Overhead**: Bounded memory footprint regardless of prompt or generation length.

---

## 3. Reference Implementation

The complete PyTorch and CUDA-compatible implementation is provided in `neuralops/models/paged_state.py` and benchmarked in `tests/test_vllm_and_sglang_plugins.py`.
