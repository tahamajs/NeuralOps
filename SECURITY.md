# NeuralOps Security Policy

## Supported Versions

Only the latest production release receives active security patches:

| Version | Supported | Notes |
| :--- | :---: | :--- |
| **2.1.x** | :white_check_mark: Active | Current active production release (Unified 3-level dispatch, 203 modules, AGT suite) |
| 2.0.x | :warning: Maintenance | Security fixes only |
| $\le$ 1.9.x | :x: End of Life | Upgrade to latest `v2.x` recommended |

---

## Reporting a Vulnerability

If you discover a potential security issue in NeuralOps (e.g. unsafe deserialization, arbitrary code execution in checkpoint loaders, buffer overflow in native CUDA extensions, memory corruption in Triton JIT kernels):

1. **Do not create a public GitHub Issue.**
2. Report the vulnerability privately via **[GitHub Private Vulnerability Reporting](https://github.com/tahamajs/NeuralOps/security/advisories/new)**.
3. Or email the core maintainer directly at: `tahamaj4@gmail.com` with the subject line: `[SECURITY] NeuralOps Vulnerability Report`.

Please include:
- A detailed description of the vulnerability and its potential impact.
- Step-by-step reproduction instructions or proof-of-concept code.
- Affected components (CUDA kernels, PyTorch modules, Serving RPC, Web endpoints).

We aim to acknowledge reports within **48 hours** and provide patches within **14 days**.

---

## Core Security & Operational Invariants

NeuralOps enforces strict security protocols across execution, deserialization, telemetry, and external integrations:

### 1. Secret Hygiene & Zero Credential Leaking
- **Environment Isolation:** Secrets such as Hugging Face tokens (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`) or API keys must never be committed to source control or logged in artifacts.
- **CLI Flag Deprecation:** CLI flags passing credentials in plaintext (e.g., `--hf-token`) are strictly prohibited and deprecated in favor of environment variables (`HF_TOKEN`).
- **Telemetry Redaction:** All diagnostic traces, model upload logs, crash dumps, and run metadata automatically scrub token patterns matching `hf_[A-Za-z0-9]{20,}` or active environment tokens via `[REDACTED_HF_TOKEN]`.
- **Zero Token Reflection:** Hugging Face Hub operations fail closed with clear warnings if credentials are unset; tokens are never printed to terminal stdout/stderr.

### 2. Fail-Closed Execution Contracts
- **Explicit CUDA Request:** When execution targets CUDA (`--device cuda` or `device="cuda"`), the system checks `torch.cuda.is_available()`. If CUDA is unavailable, the runtime immediately raises a fatal `RuntimeError` rather than silently degrading or falling back to CPU.
- **Zero Fallback to Generic SVD:** When a named compressor from the 82-compressor zoo fails or is unavailable, the compressor raises an explicit exception. Substituting generic SVD or identity matrices under a failed compressor's name is strictly forbidden.
- **Strict Numerical Parity:** If kernel computations produce non-finite values (`NaN`, `Inf`) or deviate beyond tolerance (`max_diff > tol`), execution is marked `FAIL_NUMERICS`. Silent swallowing (`except Exception: pass`) in data-producing paths is prohibited.

### 3. Untrusted Remote Code Policy (`trust_remote_code`)
- **Default Disabled:** Across all model loaders, config parsers, benchmarks, and evaluation harnesses, `trust_remote_code` defaults to `False`.
- **Audit & Recording:** Executing arbitrary Python code from remote repositories requires explicit user authorization (`--trust-remote-code` or `trust_remote_code=True`). Any invocation with remote code enabled logs an explicit audit notice recording the target model repository.
- **Safe Weight Deserialization:** Model loading enforces weights-only deserialization (`weights_only=True` via PyTorch or native `safetensors`), preventing arbitrary code execution from malicious pickled checkpoints.

### 4. Telemetry & Monitoring Localhost Binding
- **Default Loopback Binding:** Web monitors, FastAPI telemetry streaming endpoints, WebSocket feeds, and UI preview servers default to binding to `127.0.0.1` (localhost).
- **Network Exposure Protection:** Telemetry and live monitoring servers must never expose internal metrics or control endpoints to public interfaces (`0.0.0.0`) by default.
- **CORS & Input Validation:** Web interfaces enforce CORS restrictions limited to localhost origins and utilize parameterized SQL queries to prevent injection.
