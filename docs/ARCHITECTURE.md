# NeuralOps Architecture & System Design

NeuralOps is designed around clear separation of concerns across multiple layers:

```
┌────────────────────────────────────────────────────────┐
│             Pretrained Backbone (HF Models)            │
│         BERT, RoBERTa, DistilBERT, GPT-2, LLaMA        │
└───────────────────────────┬────────────────────────────┘
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
 ┌───────────────────────────┐ ┌───────────────────────────┐
 │    ATTENTION REGISTRY     │ │   COMPRESSION REGISTRY    │
 │       (130 Modules)       │ │       (73 Engines)        │
 ├───────────────────────────┤ ├───────────────────────────┤
 │ • Linear, Kernel, Cosform │ │ • ASVD, SliceGPT, Wanda   │
 │ • Sliding Window, GQA     │ │ • GPTQ, AWQ, SmoothQuant  │
 │ • SVD Factorized Attention│ │ • KV Cache Compressors    │
 └─────────────┬─────────────┘ └─────────────┬─────────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │               Triton Kernels & Hardware                 │
 │ • 19 Hand-Written Triton Kernels (GPU)                  │
 │ • 100% PyTorch CPU/MPS Fallbacks (macOS, Windows, CPU)  │
 └────────────────────────────┬────────────────────────────┘
                              ▼
 ┌─────────────────────────────────────────────────────────┐
 │             Evaluation, Benchmarks & Serving            │
 │ • Universal Sweep CLI (speed, accuracy, memory, Pareto) │
 │ • High-Throughput InferenceEngine & Continuous Batcher  │
 └─────────────────────────────────────────────────────────┘
```

## Layer Separation

1. **`core/`**: Dynamic module registries (`ATTENTION_REGISTRY`, `COMPRESSOR_REGISTRY`), Triton kernels, and architecture detection adapters.
2. **`attentions/`**: Self-contained attention mechanism implementations across 11 families.
3. **`compression_engine/`**: Post-training compression algorithms across 7 families.
4. **`benchmarks/`**: Unified CLI harness, execution orchestration, and metric aggregations.
5. **`inference/`**: High-concurrency autoregressive generation runtime with speculative decoding and continuous batching.
