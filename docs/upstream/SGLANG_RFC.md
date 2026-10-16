# [RFC] Subquadratic Recurrent State Prefix Caching for SGLang RadixAttention

- **RFC Target**: `sgl-project/sglang`
- **Author**: Taha Majlesi (`neuralops` maintainers)
- **Status**: Proposed / Upstream RFC
- **Date**: October 2026

---

## 1. Motivation

SGLang introduced RadixAttention, which maintains an in-memory Radix Tree over token sequences to reuse KV cache across multiple generation requests sharing common prefixes (e.g., system prompts, few-shot examples, multi-turn dialogues).

Standard RadixAttention stores $(K, V)$ tensor slices along radix edges. When using subquadratic linear attention models (e.g. Gated DeltaNet, RWKV-6, SVD Linear Attention), caching full sequence $(K, V)$ tokens wastes memory. Instead, caching the **fixed-dimensional cumulative recurrent state $S_t$** at prefix boundary nodes enables instantaneous $O(1)$ prefill reuse without re-scanning historical tokens.

---

## 2. Proposed Architecture: `SGLangRadixAdapter`

We propose integrating `SGLangRadixAdapter` (`neuralops.integrations.sglang.SGLangRadixAdapter`) into SGLang's radix cache manager:

```
                  Root Node [Empty]
                  /               \
       System Prompt [Tokens 1..32] -> Cached State S_32
             /                 \
Few-Shot A [Tokens 33..64]   Few-Shot B [Tokens 33..64]
 -> Cached State S_64              -> Cached State S_64
```

### Key Technical Contributions
1. **Recurrent State Node Caching**: Radix tree nodes store `recurrent_state: torch.Tensor` alongside standard token IDs.
2. **Dynamic Prefix Branching & Splitting**: Automatically splits node token arrays and computes intermediate states when new requests diverge mid-token chunk.
3. **Thread-Safe LRU Eviction**: Manages VRAM footprint through an eviction queue targeting leaf nodes, guarded by Python threading `RLock`.

---

## 3. Reference Implementation & Tests

Full production-grade implementation is located at `neuralops/integrations/sglang.py` with 7 unit tests in `tests/test_framework_serving_adapters.py`.
