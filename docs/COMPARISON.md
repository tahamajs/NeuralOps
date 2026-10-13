# Comparison with Existing Libraries

## Feature Matrix

| Feature | NeuralOps | FlashAttention | FLA | xformers | vLLM |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Attention mechanisms** | 159 | 1 (FA2/FA3) | ~30 | ~15 | ~5 |
| **Compression engines** | 73 | ❌ | ❌ | ❌ | ❌ |
| **Triton kernels** | 16 | ✅ | ✅ | ⚠️ | ✅ |
| **CPU fallback** | ✅ | ⚠️ | ✅ | ⚠️ | ❌ |
| **HF integration** | ✅ | ✅ | ✅ | ✅ | ⚠️ |
| **Benchmark CLI** | ✅ | ❌ | ✅ | ❌ | ✅ |
| **Zero-shot injection** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Model compression** | ✅ | ❌ | ❌ | ❌ | ⚠️ |
| **SVD factorization** | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Backend extras** | ✅ | ❌ | ✅ | ❌ | ❌ |
| **Documentation** | ✅ | ✅ | ✅ | ✅ | ✅ |

## Unique Features

NeuralOps is unique in combining:
1. **Massive attention library** (130 mechanisms)
2. **Complete compression suite** (73 engines)
3. **Zero-shot module injection** across 8 architectures
4. **Unified benchmark CLI**
5. **Comprehensive deprecation system** (657 aliases)

## When to use NeuralOps

- ✅ **Research on new attention mechanisms** — 159 baselines ready
- ✅ **Model compression experiments** — 73 engines ready
- ✅ **Benchmarking attention variants** — unified CLI
- ✅ **Zero-shot model surgery** — inject into any HF model

## When to use alternatives

- ⚡ **Maximum speed only** — use FlashAttention-3 directly
- 🚀 **Production inference** — use vLLM
- 🔬 **Linear attention research** — use FLA
- 🎨 **Memory-efficient attention** — use xformers
