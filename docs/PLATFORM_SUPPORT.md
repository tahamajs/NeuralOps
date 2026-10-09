# Platform Support Matrix

NeuralOps is designed to run anywhere PyTorch runs, from lightweight edge devices and laptops to multi-GPU clusters.

## Hardware & OS Compatibility

| Platform | Processor / Accelerator | PyTorch Device | Triton Acceleration | Fallback Support | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Linux (Ubuntu / RHEL)** | NVIDIA Ampere / Hopper / Blackwell | `cuda` | Yes (Full) | Native PyTorch | ✅ Fully Supported |
| **Linux (Ubuntu / RHEL)** | AMD Instinct (ROCm) | `cuda` (ROCm) | Yes | Native PyTorch | ✅ Fully Supported |
| **Linux** | x86_64 / ARM64 CPU | `cpu` | No (Not applicable) | Native PyTorch / AVX512 | ✅ Fully Supported |
| **macOS (Sonoma / Sequoia)** | Apple Silicon (M1/M2/M3/M4) | `mps` / `cpu` | No (Linux only) | Vectorized MPS & PyTorch | ✅ Fully Supported |
| **macOS** | Intel x86_64 | `cpu` | No | Native PyTorch | ✅ Supported |
| **Windows 10 / 11** | NVIDIA CUDA | `cuda` | Experimental | Native PyTorch | ✅ Supported |
| **Windows 10 / 11** | CPU | `cpu` | No | Native PyTorch | ✅ Supported |

---

## Python Version Support

- **Python 3.10**: Tested and supported.
- **Python 3.11**: Tested and supported.
- **Python 3.12**: Tested and supported.
- **Python 3.14**: Compatible with AST and runtime guards.

---

## PyTorch Version Compatibility

- **PyTorch 2.2.x**: Minimum recommended release.
- **PyTorch 2.3.x - 2.6.x**: Full feature support (SDPA, torch.compile, CUDAGraphs).
- **Transformers**: Compatible with versions 4.36 through 5.x.

---

## Triton Compatibility Notes

Triton is dynamically imported with defensive try/except guards across all kernels. When Triton is not installed or when running on non-CUDA platforms, the library emits no crashes or fatal errors and executes the numerically equivalent PyTorch implementation.
