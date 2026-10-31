# NeuralOps Security Policy

## Supported Versions

Only the latest production release receives active security patches:

| Version | Supported |
| :--- | :--- |
| **1.9.x** | :white_check_mark: Active |
| 1.8.x | :warning: Security fixes only |
| 1.7.x | :x: End of Life |
| 1.7.x | :warning: Security-fix-only |
| 1.6.x | :warning: Security-fix-only |
| $\le$ 1.5.x | :x: End of Life |

---

## Reporting a Vulnerability

If you discover a potential security issue in NeuralOps (e.g. unsafe deserialization, arbitrary code execution in checkpoint loaders, buffer overflow in native CUDA extensions, memory corruption in Triton JIT kernels):

1. **Do not create a public GitHub Issue.**
2. Report the vulnerability privately via **[GitHub Private Vulnerability Reporting](https://github.com/tahamajs/SVD_linear_Attention/security/advisories/new)**.
3. Or email the core maintainer directly at: `tahamaj4@gmail.com` with the subject line: `[SECURITY] NeuralOps Vulnerability Report`.

Please include:
- A detailed description of the vulnerability and its potential impact.
- Step-by-step reproduction instructions or proof-of-concept code.
- Affected components (CUDA kernels, PyTorch modules, Serving RPC, Web endpoints).

We aim to acknowledge reports within 48 hours and provide patches within 14 days.
