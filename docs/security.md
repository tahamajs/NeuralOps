# NeuralOps Security Policy

## Supported Versions

Only the latest production release receives active security patches:

| Version | Supported | Notes |
|---|:---:|---|
| **2.1.x / 2.0.x** | :white_check_mark: Active | Current active production release (Unified 3-level dispatch, 203 modules, AGT suite) |
| 1.9.x | :warning: Maintenance | Security fixes only |
| $\le$ 1.8.x | :x: End of Life | Upgrade to latest `v2.x` recommended |

---

## Reporting a Vulnerability

We take the security of NeuralOps and its compiled kernel execution pipeline seriously. If you discover a security vulnerability, memory corruption flaw in CUDA/Triton kernels, or model deserialization exploit, please report it responsibly:

1. **Do not create a public GitHub Issue** for undisclosed security vulnerabilities.
2. Report the vulnerability privately via **[GitHub Private Vulnerability Reporting](https://github.com/tahamajs/NeuralOps/security/advisories/new)**.
3. Or email the core maintainer directly at: **`tahamaj4@gmail.com`** with the subject line: `[SECURITY] NeuralOps Vulnerability Report`.

Please include in your report:
- Component affected (e.g., `neuralops.ops.cuda`, `neuralops.integrations.tgi`, Triton kernel memory offset).
- Minimal reproducible proof-of-concept (PoC) script.
- Operating system, PyTorch version, and CUDA / Triton driver versions.
- Potential impact (arbitrary code execution, host denial of service, memory out-of-bounds read/write).

We aim to acknowledge reports within **48 hours** and provide patches within **14 days**.

---

## Secure Model Deserialization & Serialization Policy

- **Pickle Security**: `NeuralOps` enforces strict weights-only loading (`weights_only=True` via `torch.load` and native `safetensors` format) when loading serialized model checkpoints and adapter weights.
- **Triton JIT Kernel Compilation**: Kernel generation validates compilation block sizes, tensor bounds, and power-of-2 alignment to prevent buffer overruns on GPU SRAM/L1 cache.
- **Serving Adapters**: The continuous batching adapters (`tgi`, `sglang`) enforce maximum request length limits and thread-safe lock isolation to guard against denial-of-service memory exhaustion attacks.
