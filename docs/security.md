# Security Policy

## Supported Versions

The following versions of `NeuralOps` (`SVD_linear_Attention`) receive security updates:

| Version | Supported |
|---|---|
| `1.3.x` | :white_check_mark: Yes |
| `< 1.3.0` | :x: No |

---

## Reporting a Vulnerability

We take the security of `NeuralOps` and its compiled kernel execution pipeline seriously. If you discover a security vulnerability, memory corruption flaw in CUDA/Triton kernels, or model deserialization exploit, please report it responsibly.

### How to Report

1. **Do not create a public GitHub issue** for undisclosed security vulnerabilities.
2. Email your report to: **security@neuralops.ai** or open a private advisory on GitHub Security Advisories.
3. Include the following details in your report:
   - Component affected (e.g., `neuralops.export.exporter`, `neuralops.integrations.tgi`, Triton kernel memory offset).
   - Minimal reproducible proof-of-concept (PoC) script.
   - Operating system, PyTorch version, and CUDA / Triton driver versions.
   - Potential impact (arbitrary code execution, host denial of service, memory out-of-bounds read/write).

---

## Secure Model Deserialization & Serialization Policy

- **Pickle Security**: `NeuralOps` enforces strict weights-only loading (`weights_only=True` via `torch.load` and native `safetensors` format) when loading serialized linear attention adapter weights.
- **Triton JIT Kernel Compilation**: Kernel generation validates compilation block sizes, tensor bounds, and alignment to prevent buffer overruns on GPU SRAM/L1 cache.
- **Serving Adapters**: The continuous batching adapters (`tgi.py`, `sglang.py`) enforce maximum request length limits and thread-safe lock isolation to guard against denial-of-service memory exhaustion attacks.
