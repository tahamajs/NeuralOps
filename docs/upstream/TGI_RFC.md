# [RFC] Native Linear Attention Continuous Batching Adapter for HuggingFace TGI

- **RFC Target**: `huggingface/text-generation-inference`
- **Author**: Taha Majlesi (`neuralops` maintainers)
- **Status**: Proposed / Upstream RFC
- **Date**: October 2026

---

## 1. Motivation

Text Generation Inference (TGI) provides ultra-high throughput continuous batching and token streaming for standard causal transformer models. However, when deploying compressed SVD attention models or non-softmax linear recurrent layers, standard flash-attention KV-cache prefill/decode logic incurs unnecessary cache resizing and memory allocation.

---

## 2. Proposed Architecture: `TGIServingAdapter`

We propose upstreaming the `TGIServingAdapter` protocol (`neuralops.integrations.tgi.TGIServingAdapter`) into TGI's router and generation engine:

```
[HTTP / gRPC Client] ---> [TGI Webserver / FastAPI Router]
                                    |
                                    v
                         [TGIServingAdapter]
                      (Continuous Batching Loop)
                         /          |          \
              Request 1          Request 2          Request 3
           (Active Prefill)   (Decode Token 4)   (Decode Token 19)
```

### Key Technical Contributions
1. **Dynamic Slot Management**: Handles heterogeneous prompt lengths without padding cross-talk by maintaining per-request continuous recurrent buffers.
2. **True Token-by-Token Streaming (`TGIStreamingIterator`)**: Yields decoded text chunks immediately as individual tokens are sampled, adhering to the Server-Sent Events (SSE) format expected by Hugging Face Hub clients.
3. **Stop Sequence & Repetition Penalty Enforcement**: High-efficiency in-flight checking against user-specified stop tokens and max sequence limits.

---

## 3. Reference Implementation & Tests

Reference implementation resides at `neuralops/integrations/tgi.py` with 7 dedicated unit tests verified in `tests/test_framework_serving_adapters.py`.
