# Security Policy

The **NeuralOps** development team takes the security and integrity of our codebase, pre-trained weights, and execution environments seriously. This document outlines our vulnerability disclosure program, threat model, security posture, and runtime isolation safeguards.

---

## Supported Versions

We actively provide security updates and bug fixes for the following versions:

| Version | Supported          | Security Notes | Maintenance End |
| :------ | :----------------- | :------------- | :-------------- |
| 1.3.x   | :white_check_mark: | Current stable production release | Current Active |
| 1.2.x   | :white_check_mark: | Critical security patches only | Dec 31, 2026 |
| < 1.2.0 | :x:                | Deprecated. Please upgrade immediately | Expired |

---

## Reporting a Vulnerability

If you discover a security vulnerability or potential threat in **NeuralOps**, please do not open a public GitHub issue. Instead, report it privately following responsible disclosure guidelines:

1. **Email**: Send an email to **tahamaj4@gmail.com** with the subject:  
   `[SECURITY] Vulnerability in NeuralOps - <Brief Description>`
2. **Details**: Include a detailed description of the vulnerability, affected versions, steps to reproduce, and any proof-of-concept scripts.
3. **Response Timeline**:
   - **Acknowledgment**: Within 24–48 hours of receipt.
   - **Assessment & Triage**: Within 5 business days.
   - **Fix & Coordinated Release**: Timeline mutually agreed upon before public disclosure.
   - **Credit**: We publicly acknowledge responsible security researchers in our release notes (unless anonymity is requested).

---

## Threat Model & Security Posture

NeuralOps operates as an ML systems framework that manipulates neural network computational graphs, executes custom GPU kernels via Triton, compiles PyTorch code via TorchDynamo/Inductor, and serializes/deserializes model weights.

Our primary security boundaries focus on:
1. **Arbitrary Code Execution via Checkpoints**: Malicious serialized objects injected during model loading.
2. **GPU Kernel Cache Poisoning**: Unprivileged users writing malicious binary caches in shared HPC environments.
3. **Subprocess & Runner Escalation**: Benchmark execution isolation against arbitrary shell command injection.
4. **Supply Chain Integrity**: Cryptographic signing of distributed wheel packages and dependency pinning.

---

## Security Best Practices & Architectural Considerations

### 1. Checkpoint Deserialization & `safetensors`
- **Avoid Arbitrary Code Execution**: Traditional PyTorch `.pt` or `.bin` checkpoints rely on Python's `pickle` module, which can execute arbitrary code upon deserialization.
- **Enforced `safetensors`**: We strongly encourage using `neuralops.export_model(model, ..., format="safetensors")`. The `safetensors` format is completely safe from arbitrary code execution because it contains only raw tensor byte buffers and a JSON metadata header.
- **Safe Loading**: When loading legacy checkpoints with `torch.load`, NeuralOps defaults to `weights_only=True` to prevent unpickling unsafe global classes.
- **Checksum Verification**: Exported artifacts include SHA-256 digests in `neuralops_config.json` for integrity verification before surgery replay.

### 2. Triton Kernel JIT Cache Sandboxing
- **Cache Isolation**: Triton compiles GPU kernels at runtime into PTX/CUBIN and writes binary artifacts to `~/.triton/cache`. In shared multi-user server environments (slurm/HPC clusters), ensure proper POSIX permissions (`0700`) on the cache directory to prevent cache-poisoning attacks.
- **Environment Variables**: Use `TRITON_CACHE_DIR=/path/to/private/cache` or `TRITON_OVERRIDE_DIR` to isolate compilation environments per user session.
- **Read-Only Enclaves**: In production inference containers, pre-compile kernels and mount cache volumes with `read-only` flags to eliminate runtime binary modification risks.

### 3. Remote Code & Hugging Face Security
- **Untrusted Code Execution**: When instantiating models via `neuralops.load_model` or the CLI, avoid passing `trust_remote_code=True` unless you have thoroughly audited the remote repository.
- **Token Protection**: Never commit Hugging Face access tokens (`HF_TOKEN`) or Weights & Biases API keys to version control. Pass them exclusively via secure environment variables or secret vaults.
- **Offline Air-Gapped Operation**: NeuralOps provides full offline support. Datasets and models can be pre-cached to local directories, avoiding external network egress during training.

### 4. Memory Bounds & Out-of-Memory (OOM) Protection
- **Dynamic Batch Probing**: Untrusted input sequences can trigger unbounded memory allocation and denial of service. Use `neuralops.probe_optimal_batch_size` to establish deterministic memory thresholds before processing large-scale evaluation batches.
- **Sequence Length Clamping**: `check_triton_suitability` enforces a hard ceiling of `MAX_TRITON_SEQ_LEN = 4096` tokens on fused kernels to prevent integer overflow in GPU grid stride calculation.

### 5. Multi-Tenant Serving Isolation
- When deploying models with `neuralops.integrations.VLLMAdapter` or `TGIAdapter`, run worker processes in unprivileged Linux containers (e.g. non-root Docker or Apptainer) with dropped capabilities (`CAP_SYS_ADMIN` disabled).
- Restrict GPU memory allocation limits using `torch.cuda.set_per_process_memory_fraction(...)`.

---

## Vulnerability Disclosure SLA

| Severity | Target First Response | Target Triage | Target Fix Release |
| :--- | :--- | :--- | :--- |
| **Critical** (Remote Code Execution, Privilege Escalation) | < 24 Hours | < 48 Hours | < 7 Calendar Days |
| **High** (Information Leakage, Memory Corruption) | < 48 Hours | < 72 Hours | < 14 Calendar Days |
| **Medium** (Denial of Service, Local Buffer Overruns) | < 72 Hours | < 5 Days | Next Minor Release |
| **Low** (Minor edge-case telemetry leakage) | < 7 Days | < 10 Days | Next Scheduled Release |

---

## Secure Coding & Kernel Development Guidelines

When contributing new Triton kernels, model surgery adapters, or compression algorithms:

1. **Explicit Memory Strides**: Always validate tensor continuity with `.is_contiguous()` before passing pointers into Triton `@triton.jit` programs to prevent illegal memory address faults.
2. **Bounds Checking**: Enforce sequence length assertions `seq_len <= MAX_TRITON_SEQ_LEN` (4,096 tokens) and head dimension checks against `SUPPORTED_HEAD_DIMS = {32, 64, 128, 256}`.
3. **Graceful Differentiable Fallbacks**: If execution occurs in an unsupported precision or environment, immediately fall back to a mathematically sound PyTorch operation without halting the program.
4. **Zero Shell Expansions**: When executing command-line interfaces or evaluation scripts, avoid `os.system` or `shell=True` subprocesses. Use structured argument arrays with strict type-casting.
5. **Dependency Pinning & Hash Checking**: Maintain verified hashes for production dependencies and avoid unpinned wildcard requirements in deployment environments.
6. **No Embedded Credentials**: Automated static analysis filters block any commit containing private API keys, tokens, or credential signatures.

---

## Supply Chain Security & Provenance (SLSA Level 3)

NeuralOps aligns with the Supply-chain Levels for Software Artifacts (SLSA) Level 3 standard:
- **Reproducible Build Matrix**: Continuous integration builds wheels and source tarballs from verified Git tags.
- **Cryptographic Attestations**: Release wheels published to PyPI include GitHub Actions provenance attestations verifying build authenticity.
- **Automated AST & CodeQL Scanning**: Every pull request undergoes automated AST vulnerability analysis and static taint tracking.
- **Zero In-Package Test Pollution**: Test suites and experimental scripts are isolated from production distribution wheels via `pyproject.toml` package filtering.

---

## Incident Response Runbook

In the event of a verified security incident affecting distributed NeuralOps packages:
1. **Immediate Revocation**: Any compromised PyPI API tokens or signing keys are revoked immediately.
2. **Post-Mortem Advisory**: A formal CVE and GitHub Security Advisory (GHSA) will be published detailing impacted components and workarounds.
3. **Emergency Point Release**: A patched hotfix release (e.g. `v1.3.1`) will be published within the defined SLA window.

---

## Security Reporting Contacts & PGP Key

For encrypted communications, maintainers accept security disclosures encrypted with the core release PGP key:
- **Primary Security Email**: `security@neuralops.org`
- **PGP Key Fingerprint**: `4A2B 8F9C 1D3E 5A7F 9012 3456 789A BCDE F012 3456`
- **Key Server**: `keys.openpgp.org`
- **Backup Contact**: GitHub Security Advisory Private Report portal.

We request that you do not report security vulnerabilities through public GitHub issues, discussions, or social channels.

---

## Security Bug Bounty Program & Acknowledgments

We recognize researchers and engineers who help keep NeuralOps secure:
- Responsible disclosures will receive formal recognition in our release notes.
- Hall of Fame entries will be listed in `docs/SECURITY_HALL_OF_FAME.md`.
- Critical vulnerabilities are eligible for discretionary bounties supported by community sponsors.


