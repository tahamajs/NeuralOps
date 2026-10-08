# 6. Experimental Setup and Evaluation Protocol

## 6.1 Benchmark Datasets and Tasks
Empirical evaluations follow rigorous standardized protocols across core natural language benchmarks:
- **SST-2 (Stanford Sentiment Treebank)**: Binary sentiment classification evaluating sentence-level representation fidelity under parameter reduction.
- **MRPC (Microsoft Research Paraphrase Corpus)**: Semantic sentence-pair equivalence assessing cross-sentence contextual encoding.
- **GLUE / Perplexity Sweeps**: Evaluation of generative token loss and perplexity on autoregressive architectures.

| Dataset | Split Size | Sequence Length | Evaluation Metric |
|---|---|---|---|
| SST-2 | 67,349 Train / 872 Val | 128 | Accuracy (%) |
| MRPC | 3,668 Train / 408 Val | 128 | F1 Score / Accuracy |
| Wikitext-2 | 2.0M Tokens | 512 | Perplexity |

## 6.2 Target Model Backbones
Evaluations span four diverse architectural paradigms:
1. **DistilBERT-base-uncased**: 6 layers, 12 heads, 768 hidden dimension (66.9M parameters), serving as a standard flat bidirectional encoder.
2. **RoBERTa-base**: 12 layers, 12 heads, 768 hidden dimension (124.6M parameters), representing deep nested encoder architectures.
3. **GPT-2**: 12 layers, 12 heads, 768 hidden dimension (124.4M parameters), representing causal autoregressive decoders.
4. **Qwen 3.5 / LLaMA-style**: Modern generative architectures featuring Grouped-Query Attention (GQA) and RoPE embeddings.

## 6.3 Hardware Environment and Metrics Collection
All benchmarks were conducted on dedicated NVIDIA GPU accelerators running CUDA 12 and PyTorch 2.4. Peak VRAM utilization is captured via `torch.cuda.max_memory_allocated()`, and throughput is tracked in tokens per second under standardized batch sizing ($B=64$, sequence length $L=128$).
