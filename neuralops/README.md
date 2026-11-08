# 🧬 NeuralOps Core Package (`neuralops`)

NeuralOps v2.1.0 is the high-performance core Python engine of the NeuralOps ecosystem, purpose-built for subquadratic attention research, in-place neural architecture surgery, mathematical model compression, and hardware-accelerated operator execution across NVIDIA GPUs, Apple Silicon Metal (MPS), and modern multicore CPUs.

This directory contains the core library source tree.

## Documentation & References
For full architecture documentation, empirical benchmark reports, and installation guides, please refer to:
- [NeuralOps Root Documentation](../README.md)
- [Comprehensive API & Architecture Docs](../docs/)
- [Examples Suite (35 Demonstration Scripts)](../examples/README.md)
- [Official GitHub Repository](https://github.com/tahamajs/NeuralOps)

## Invariant Registries
The canonical mathematical and architectural registries are maintained dynamically in the library:
- Attention Mechanisms: `neuralops.layers.ATTENTION_REGISTRY` (verified via `tests/test_registry_counts.py`)
- Compression Engines: `neuralops.compression.COMPRESSOR_REGISTRY` (verified via `tests/test_registry_counts.py`)
- Triton Kernel Catalog: `neuralops.ops.triton.TRITON_KERNEL_CATALOG`

## License
NeuralOps is licensed under the MIT License. See [LICENSE](../LICENSE) for details.
