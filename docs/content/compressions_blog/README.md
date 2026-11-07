# NeuralOps Model Compression: Blog & Technical Monographs

> [!NOTE] **Empirical Methodology & Theoretical Projection Notice**
> Benchmark metrics and trade-off tables presented in these architectural monographs reflect theoretical asymptotic scaling, design projections, or illustrative simulations from reference literature. For physical hardware benchmarks executed on dedicated silicon (NVIDIA RTX 4070 SM 8.9 / RTX 3090 SM 8.6) with cryptographic execution provenance, consult the authoritative [Master Results Matrix](https://huggingface.co/datasets/Hooshaai/neuralops-results) and [Results Ledger](../../api/results_ledger.md).

Comprehensive guides covering the 72 neural compression engines in NeuralOps.

- **Activation-Aware SVD (HNC-SVD & SVD-LLM)**: Covariance whitening for zero-degradation weight truncation.
- **Structured Pruning & Sparsity**: 2:4 hardware sparsity and ADMM channel pruning.
- **Microscaled Low-Precision Quantization**: FP8 (e4m3fn/e5m2) and MX-FP4 formats.
- **KV-Cache Memory Dynamics**: Heavy Hitter Oracle (H2O) and StreamingLLM attention sinks.
