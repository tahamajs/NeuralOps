# NeuralOps Security & Operational Policy

## Supported Versions

Only the latest production release receives active security patches:

| Version | Supported | Notes |
|---|:---:|---|
| **2.1.x** | :white_check_mark: Active | Current active production release (Unified 3-level dispatch, 203 modules, AGT suite) |
| 2.0.x | :warning: Maintenance | Security fixes only |
| $\le$ 1.9.x | :x: End of Life | Upgrade to latest `v2.x` recommended |

---

## Reporting a Vulnerability

We take the security of NeuralOps, its compiled kernel execution pipeline, and serving infrastructure seriously. If you discover a security vulnerability, memory corruption flaw in CUDA/Triton kernels, or model deserialization exploit, please report it responsibly:

1. **Do not create a public GitHub Issue** for undisclosed security vulnerabilities.
2. Report the vulnerability privately via **[GitHub Private Vulnerability Reporting](https://github.com/tahamajs/NeuralOps/security/advisories/new)**.
3. Or email the core maintainer directly at: **`tahamaj4@gmail.com`** with the subject line: `[SECURITY] NeuralOps Vulnerability Report`.

Please include in your report:
- Component affected (e.g., `neuralops.ops.cuda`, `neuralops.integrations.vllm`, Triton kernel memory offset).
- Minimal reproducible proof-of-concept (PoC) script.
- Operating system, PyTorch version, and CUDA / Triton driver versions.
- Potential impact (arbitrary code execution, host denial of service, memory out-of-bounds read/write).

We aim to acknowledge reports within **48 hours** and provide patches within **14 days**.

---

## Security Architecture & Operational Invariants

NeuralOps implements strict safeguards across execution environments, remote model hubs, kernel compilers, and telemetry services.

### 1. Secret Hygiene & Zero Credential Leaks
- **Zero Plaintext Tokens:** API keys and credentials (such as Hugging Face tokens) must never be passed via command-line arguments or checked into repository manifests. Command-line flags accepting plaintext tokens are rejected.
- **Environment Resolution:** Hub integrations source authentication strictly via the `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) environment variable.
- **Automated Scrubbing:** All log handlers, diagnostic dumps, and model upload summaries automatically redact secrets matching `hf_[A-Za-z0-9]{20,}` or matching the contents of `HF_TOKEN` with `[REDACTED_HF_TOKEN]`.
- **Fail Closed on Missing Auth:** Hub upload daemons refuse unauthenticated operations without leaking credential state to terminal outputs or shared ledgers.

### 2. Fail-Closed Execution Contracts
- **Explicit CUDA Request:** When `--device cuda` or a CUDA backend is requested, the system verifies `torch.cuda.is_available()`. If CUDA is missing, execution terminates immediately with a `RuntimeError`—silent fallback to CPU or simulated metrics is strictly forbidden.
- **Zero Generic SVD Fallback:** Each compressor in the 82-compressor zoo must execute its authentic algorithm. If a named compressor fails or encounters numerical instability, it raises a terminal error; substituting generic SVD, truncated zero tensors, or identity matrices under that compressor's label is prohibited.
- **Strict Numerical Parity Gates:** Any kernel producing non-finite tensors (`NaN`, `Inf`) or exceeding error thresholds (`max_diff > tolerance`) is designated `FAIL_NUMERICS`. Catching exceptions in data-producing paths with `except Exception: pass` is prohibited.
- **Zero Fake Artifacts:** Never write synthetic metrics, unverified throughput numbers, or placeholder binary files when real evaluation fails.

### 3. Remote Code Execution Policy (`trust_remote_code`)
- **Strict Default `False`:** Across all model loaders (`neuralops.benchmarks.core.model_loader`, `neuralops.models`, `neuralops.evaluation`), `trust_remote_code` defaults to `False`.
- **Explicit Opt-In Required:** Executing custom model code from remote repositories requires explicit user consent via `--trust-remote-code` or programmatic `trust_remote_code=True`.
- **Audit Logging:** Every load operation that enables `trust_remote_code` logs an explicit audit warning detailing the target repository identifier.
- **Weights-Only Deserialization:** Serialized checkpoints enforce `weights_only=True` via `torch.load` and prefer native SafeTensors (`safetensors`) to block arbitrary pickle deserialization exploits.

### 4. Telemetry & Monitoring Localhost Binding
- **Hermetic Localhost Binding:** Live GPU telemetry servers, WebSocket monitors, dashboard runners (`run_monitor_server.py`, `neuralops.runners.monitor`), and UI preview services default to binding to `127.0.0.1` (localhost).
- **No Unintended Network Exposure:** Public interface exposure (`0.0.0.0`) requires explicit operator configuration; services never expose live metrics, system traces, or execution controls to external networks by default.
- **CORS & Input Validation:** Web telemetry endpoints enforce strict localhost CORS origins (`127.0.0.1:8080`, `localhost:8080`) and sanitize query parameters against SQL injection via parameterized statements.
