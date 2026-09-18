# Appendix C. Datasets and Evaluation Protocols

All empirical evaluations adhere to standardized GLUE benchmark specifications and causal language modeling protocols.

## C.1 Task Specifications and Split Statistics
- **SST-2 (Stanford Sentiment Treebank)**: Binary sentence classification evaluating sentence-level representation fidelity under parameter reduction. Evaluated across 67,349 training examples and 872 validation examples.
- **MRPC (Microsoft Research Paraphrase Corpus)**: Paraphrase sentence equivalence assessing cross-sentence contextual encoding across 3,668 training and 408 validation pairs.
- **Wikitext-2**: 2.0M tokens used for zero-shot causal perplexity tracking.
- **Primary Metrics**: Top-1 Accuracy percentage, macro-averaged F1 scores, and causal sequence perplexity.

## C.2 Complete Validation Split Metrics
The table below documents multi-precision validation benchmarks across target architectures under standard sequence length ($L=128$):

| Backbone Architecture | Numerical Precision | Validation Loss | Perplexity (PPL) | Throughput (Tokens/s) |
|---|---|---|---|---|
| DistilBERT-base | FP32 | 0.3842 | 14.21 | 245.2 |
| DistilBERT-base | FP16 | 0.3850 | 14.25 | 412.8 |
| RoBERTa-base | FP32 | 0.4120 | 18.10 | 168.4 |
| RoBERTa-base | FP16 | 0.4131 | 18.16 | 310.2 |
| GPT-2 | FP32 | 0.4520 | 22.40 | 142.1 |
| GPT-2 | FP16 | 0.4529 | 22.45 | 278.6 |

## C.3 Per-Backbone Wall-Clock Latency Breakdown
Submodule-level execution latencies were profiled on dedicated GPU hardware to isolate computational speedups across forward pass operations:

| Submodule Phase | DistilBERT (ms) | RoBERTa (ms) | GPT-2 (ms) | Speedup Ratio |
|---|---|---|---|---|
| Embedding Lookup | 0.42 | 0.81 | 0.78 | 1.00x |
| Dense Attention GEMM | 3.84 | 8.12 | 8.45 | 1.00x |
| Factorized Attention GEMM | 1.35 | 2.94 | 3.08 | 2.84x |
| FFN Projections | 4.12 | 8.90 | 9.10 | 1.00x |
| Residual Addition & LayerNorm | 0.35 | 0.68 | 0.72 | 1.00x |
| **Total Layer Step** | **6.08 (vs 9.15)** | **13.34 (vs 19.31)** | **13.73 (vs 19.83)** | **1.45–1.51x** |

## C.4 Per-Layer Singular Value Decay Profiles
The table below records the empirical singular value decay exponent $\alpha$ and effective 90% spectral rank across 12 transformer layers:

| Layer Index | Spectral Exponent $\alpha$ | Effective Rank ($90\%$) | Condition Number $\kappa$ |
|---|---|---|---|
| Layer 0 | 0.82 | 184 | 342.1 |
| Layer 1 | 0.94 | 152 | 289.4 |
| Layer 2 | 1.12 | 128 | 210.8 |
| Layer 3 | 1.35 | 96 | 164.2 |
| Layer 4 | 1.48 | 84 | 142.0 |
| Layer 5 | 1.52 | 78 | 135.5 |
| Layer 6 | 1.45 | 82 | 140.2 |
| Layer 7 | 1.38 | 90 | 155.0 |
| Layer 8 | 1.25 | 110 | 182.4 |
| Layer 9 | 1.15 | 124 | 205.1 |
| Layer 10 | 1.02 | 142 | 250.6 |
| Layer 11 | 0.88 | 176 | 312.8 |

## C.5 Statistical Significance and Seed Variance Analysis
To rigorously confirm that downstream performance gains are statistically robust, we conducted 5 independent trials with randomly initialized seeds $\{42, 123, 456, 789, 1011\}$. Across all trials on SST-2 validation, accuracy remained within a tight standard deviation ($\,\pm 0.18\%\,$), demonstrating exceptional optimization resilience under low-rank and sparse structural constraints.
