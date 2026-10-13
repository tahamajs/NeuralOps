# Security Policy

## Supported Versions

We actively provide security updates for the following versions of `NeuralOps`:

| Version | Supported          |
| ------- | ------------------ |
| 1.3.x   | :white_check_mark: |
| 1.2.x   | :white_check_mark: |
| < 1.2.0 | :x:                |

---

## Reporting a Vulnerability

If you discover a security vulnerability in `NeuralOps`, please report it privately rather than opening a public GitHub issue:

1. Send an email to **tahamaj4@gmail.com** with the subject `[SECURITY] Vulnerability in NeuralOps`.
2. Include a clear description of the issue, affected versions, a minimal reproduction script, and potential impact.
3. We will acknowledge receipt within 48 hours and work with you on a coordinated disclosure timeline and fix release.

---

## Security Considerations

### 1. Deserialization & Untrusted Checkpoints
- **Always use `safetensors`**: When loading external weights or exporting models with `neuralops.export_model`, prefer the `"safetensors"` format over legacy PyTorch `.pt` / `.bin` pickle files. Pickled checkpoints can execute arbitrary code upon deserialization.
- **`torch.load` safety**: Never pass `weights_only=False` to `torch.load` when loading untrusted model checkpoints.

### 2. Triton Kernel JIT Compilation Cache
- By default, Triton caches compiled kernel binaries in `~/.triton/cache`. In shared or multi-tenant server environments, ensure appropriate file system permissions (`0700`) on the cache directory to prevent cache poisoning attacks.

### 3. Hugging Face `trust_remote_code`
- When executing benchmarks via `neuralops run`, `trust_remote_code` is disabled by default if the `--no-trust-remote-code` flag is specified. Only load custom model code from trusted authors.
