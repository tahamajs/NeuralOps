# Verification Rationale & Non-Bug Determinations (Waves 1–4)

This document records technical forensic investigations where an automated scan, linter flag, or preliminary review flagged an issue, but deep technical audit determined that the current behavior is intentional, correct, or mathematically required ("Not a Bug").

---

## 1. Mathematical & Numerical Determinations

### Case 1: `rfa_attention` — Absence of Artificial Dynamic Range Clamping
- **Flagged Symptom:** Proposed clamping intermediate state accumulator $s$ and denominator to $\pm 65504$ (FP16 ceiling).
- **Forensic Investigation:**
  - The PyTorch reference implementation (`neuralops/ops/cuda/rfa_attention/rfa_attention_autograd.py`) performs state accumulation in full IEEE-754 single precision (`float32`), computing $s_{t} = s_{t-1} + \phi(k_t) v_t$.
  - Introducing an artificial clamp at 65504 truncated valid intermediate sums when sequence lengths exceeded 2048, inducing artificial divergence from PyTorch reference.
- **Determination:** **NOT A BUG**. Accumulation in FP32 with non-finite guards (`isnan || isinf ? 0 : x`) is mathematically sound and matches PyTorch autograd semantics.

### Case 2: `phantom` — Rejection of Fast-Math Intrinsics
- **Flagged Symptom:** Discrepancy observed at Heaviside gate threshold ($g \approx \tau$); proposed using `__fdividef` and `__expf` fast-math instructions.
- **Forensic Investigation:**
  - Fast-math intrinsics relax IEEE-754 rounding rules, introducing 2–3 ULP errors into sigmoid computation.
  - The observed test discrepancy occurred precisely at the step-function boundary where infinitesimal rounding differences between CPU and GPU hardware place values on opposite sides of $g \ge \tau$.
- **Determination:** **NOT A BUG**. Standard precision operations must be preserved; threshold boundary variance is handled via standard step-function boundary tolerance, not precision degradation.

### Case 3: `spectra_v2` — Preservation of Symplectic Ridge Regularization
- **Flagged Symptom:** Proposed eliminating small positive ridge diagonal damping $\epsilon \cdot I$ in singular value decomposition.
- **Forensic Investigation:**
  - For ill-conditioned Gram matrices, unregularized singular decomposition results in numerical instability, gradient explosion, and loss of Hamiltonian symplectic structure.
  - Adding a minimal Tikhonov ridge factor ($\epsilon = 10^{-6}$) is mathematically necessary to guarantee positive semi-definiteness during online state transitions.
- **Determination:** **NOT A BUG**. Ridge damping is an essential mathematical stability guardrail.

---

## 2. Architecture & Design Determinations

### Case 4: `tiny_gpt2_dummy` & Synthetic Test Fixtures
- **Flagged Symptom:** Flagged under Rule 2.5 / Rule 5.4 as "fake model implementation".
- **Forensic Investigation:**
  - `tiny_gpt2_dummy` and synthetic model generators in `tests/conftest.py` exist specifically to satisfy AGENTS.md Rule 5.4: *"Forbidden in tests: Real model downloads — use tiny synthetic AutoModel"*.
  - They provide deterministic, sub-10ms forward passes for unit tests without requiring Hugging Face Hub network connectivity or large disk allocations in resource-constrained CI environments.
- **Determination:** **NOT A BUG**. Synthetic test fixtures are standard engineering practice for fast, offline, isolated unit tests.

### Case 5: CUDA-Only Benchmark Status `SKIPPED` on Non-CUDA Hosts
- **Flagged Symptom:** Benchmark sweeps on CPU-only machines report `status: SKIPPED` for CUDA C++ kernels, flagged as a potential execution failure.
- **Forensic Investigation:**
  - AGENTS.md Rule 2.5 explicitly mandates: *"Never fail hard on non-CUDA hardware unless the benchmark is explicitly CUDA-only. Return BenchmarkResult(status=BenchmarkStatus.SKIPPED, reason='CUDA unavailable')"*.
  - Hardware kernels requiring CUDA or Triton cannot physically execute on Darwin/macOS or CPU hosts without hardware instruction support.
- **Determination:** **NOT A BUG**. Returning `BenchmarkStatus.SKIPPED` is the strictly enforced contract requirement.

### Case 6: Fail-Closed `IntegrationNotImplementedError` for Absent Optional Backends
- **Flagged Symptom:** Calling serving adapters (`vLLM`, `SGLang`, `TensorRT-LLM`) raises `IntegrationNotImplementedError` when optional dependencies are not installed in the environment.
- **Forensic Investigation:**
  - Rather than returning fake mocked responses or dummy tokens (which violate benchmark integrity), the framework adheres to the "Fail-Closed" invariant (Rule 9.4).
  - When optional runtime SDKs are absent, the integration layer immediately halts with a descriptive error specifying the required extra (`pip install neuralops[serving]` or `[vllm]`).
- **Determination:** **NOT A BUG**. Fail-closed rejection is required to maintain zero-fake benchmark provenance.
