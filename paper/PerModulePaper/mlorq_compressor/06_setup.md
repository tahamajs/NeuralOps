# 6. Experimental Setup

## 6.1 Benchmark Protocol
Empirical evaluations are conducted under standardized benchmark configurations across representative transformer families:
- **DistilBERT**: Standard 6-layer bidirectional encoder on GLUE SST-2.
- **RoBERTa-base**: 12-layer bidirectional representation on SST-2 and MRPC.
- **GPT-2**: 12-layer causal language model evaluated on sequence classification and perplexity.
- **Qwen 3.5 / T5**: Scaled generative backbones evaluating parameter efficiency and cross-attention stability.

## 6.2 Hardware and Environment
All benchmarks are executed on dedicated NVIDIA GPU accelerators using PyTorch with CUDA 12 backend. Peak VRAM utilization is tracked via `torch.cuda.max_memory_allocated()`, and latency metrics capture p95 forward-backward step timings.
