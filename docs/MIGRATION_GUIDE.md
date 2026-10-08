# Migration Guide: v1.0.0 -> v1.1.0

## What's New in v1.1.0
- **Modern Attention Mechanisms**: Native support for sliding window attention, ALiBi, Attention Sinks, MQA, Cross Attention, and softcapping.
- **Inference Runtime**: High-throughput `NeuralOpsInferenceEngine`, continuous batching, streaming, and speculative decoding.
- **Compression Enhancements**: Exact GPTQ, AWQ, SmoothQuant, Wanda, and DoRA.
- **Framework Integrations**: HuggingFace `AutoNeuralOpsModel`, vLLM, and TGI adapters.

## Upgrading
```bash
pip install --upgrade neuralops
```
All existing 658 deprecated aliases remain backward-compatible.
