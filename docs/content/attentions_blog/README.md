# NeuralOps Attention Mechanisms: Blog & Theoretical Monographs

> [!NOTE] **Empirical Methodology & Theoretical Projection Notice**
> Benchmark metrics and trade-off tables presented in these architectural monographs reflect theoretical asymptotic scaling, design projections, or illustrative simulations from reference literature. For physical hardware benchmarks executed on dedicated silicon (NVIDIA RTX 4070 SM 8.9 / RTX 3090 SM 8.6) with cryptographic execution provenance, consult the authoritative [Master Results Matrix](https://huggingface.co/datasets/Hooshaai/neuralops-results) and [Results Ledger](../../api/results_ledger.md).

This series explores the mathematical taxonomy, kernel acceleration, and hardware profiling across the 131 registered attention architectures in NeuralOps.

- **Spectral & SVD Low-Rank Decompositions**: Exact state factorizations and Eckart-Young-Mirsky bounds.
- **Kernelized & Feature-Map Linear Attentions**: Positive kernel mapping and Taylor approximations.
- **Continuous-Time & State-Space Recurrences**: Mamba selective scans and hardware SRAM tiling.
- **Hybrid Multi-Branch Attention**: Blending softmax precision with recurrent linear scaling.
