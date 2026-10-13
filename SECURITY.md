# Security Policy

The **NeuralOps** development team takes the security and integrity of our codebase, pre-trained weights, and execution environments seriously.

---

## Supported Versions

We actively provide security updates and bug fixes for the following versions:

| Version | Supported          | Security Notes |
| :------ | :----------------- | :------------- |
| 1.3.x   | :white_check_mark: | Current stable production release |
| 1.2.x   | :white_check_mark: | Critical security patches only |
| < 1.2.0 | :x:                | Deprecated. Please upgrade immediately |

---

## Reporting a Vulnerability

If you discover a security vulnerability or potential threat in **NeuralOps**, please do not open a public GitHub issue. Instead, report it privately following responsible disclosure guidelines:

1. **Email**: Send an email to **tahamaj4@gmail.com** with the subject:  
   `[SECURITY] Vulnerability in NeuralOps - <Brief Description>`
2. **Details**: Include a detailed description of the vulnerability, affected versions, steps to reproduce, and any proof-of-concept scripts.
3. **Response Timeline**:
   - **Acknowledgment**: Within 48 hours of receipt.
   - **Assessment & Triage**: Within 5 business days.
   - **Fix & Coordinated Release**: Timeline mutually agreed upon before public disclosure.

---

## Security Best Practices & Architectural Considerations

### 1. Checkpoint Deserialization & `safetensors`
- **Avoid Arbitrary Code Execution**: Traditional PyTorch `.pt` or `.bin` checkpoints rely on Python's `pickle` module, which can execute arbitrary code upon deserialization.
- **Enforced `safetensors`**: We strongly encourage using `neuralops.export_model(model, ..., format="safetensors")`. The `safetensors` format is completely safe from arbitrary code execution.
- **Safe Loading**: When loading legacy checkpoints with `torch.load`, always set `weights_only=True`.

### 2. Triton Kernel JIT Cache Sandboxing
- **Cache Isolation**: Triton compiles GPU kernels at runtime and writes binary artifacts to `~/.triton/cache`. In shared multi-user server environments, ensure proper POSIX permissions (`0700`) on the cache directory to prevent cache-poisoning attacks.
- **Environment Variables**: Use `TRITON_CACHE_DIR=/path/to/private/cache` to isolate compilation environments.

### 3. Remote Code & Hugging Face Security
- **Untrusted Code Execution**: When instantiating models via `neuralops.load_model` or the CLI, avoid passing `trust_remote_code=True` unless you have thoroughly audited the remote repository.
- **Token Protection**: Never commit Hugging Face access tokens (`HF_TOKEN`) or Weights & Biases API keys to version control. Pass them exclusively via secure environment variables.

### 4. Memory Bounds & Out-of-Memory (OOM) Protection
- **Dynamic Batch Probing**: Untrusted input sequences can trigger unbounded memory allocation. Use `neuralops.evaluation.probe_optimal_batch_size` to establish deterministic memory thresholds before processing large-scale evaluation batches.
