# NeuralOps Architecture Reference (v1.1.0)

NeuralOps is organized into decoupled layers:

1. **`core`**: Base attention/compressor abstractions, dynamic registry (`ATTENTION_REGISTRY`, `COMPRESSOR_REGISTRY`), Rotary embeddings, GQA utilities, numerical stability tools, and Triton GPU kernels with PyTorch fallback dispatch.
2. **`attentions`**: 145 registered attention mechanism modules across families (dense, linear, sparse, modern, ssm, svd, adaptive, fast_kernels, hybrid, gated).
3. **`compression_engine`**: 73 model compression techniques (quantization, pruning, SVD factorization, low-rank adaptations including GPTQ, AWQ, SmoothQuant, Wanda, QLoRA, DoRA).
4. **`inference`**: Continuous batcher, paged KV cache, speculative decoder, streaming generator, and inference engine.
5. **`integrations`**: Direct connectors for HuggingFace AutoModel, vLLM, TGI, DeepSpeed, and PyTorch dynamic compilation.
6. **`training`**: Selective gradient checkpointing, MixedPrecisionTrainer with gradient accumulation, and Curriculum learning.
7. **`benchmarks`**: Comprehensive speed, memory, accuracy, and scaling evaluation harnesses.
