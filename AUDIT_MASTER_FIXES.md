# Master Audit Fixes & Remediation Ledger (Waves 1–4)

This ledger comprehensively maps all bug IDs, security issues, and architectural defects identified and remediated across Wave 1, Wave 2, Wave 3, and Wave 4 of NeuralOps development.

---

## 1. High-Priority Architectural & Security Fixes (P0 / Critical)

| Bug ID | Subsystem / Component | Issue Description | Fix Applied | Remediation Commit / Wave | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SEC-01** | `benchmarks/core/paths.py`, `saver.py` | Unsafe deserialization in `torch.load` missing `weights_only=True` (CVE risk) | Mandated `weights_only=True` across all deserialization call sites | Wave 1 (`48f3281`) | **RESOLVED** |
| **SEC-02** | `benchmarks/core/secret_scanner.py` | Insufficient secret scanning in telemetry emission | Expanded redaction to 8 credential patterns (HF, OpenAI, AWS, GitHub PAT, GCP, PyPI) | Wave 1 (`b83491f`) | **RESOLVED** |
| **P0-AMP** | `ops/`, `trainer/`, `benchmarks/` | Deprecated `torch.cuda.amp` imports causing warnings and deprecation failures | Standardized on `torch.amp.autocast("cuda")` and `torch.amp.GradScaler("cuda")` | Wave 2 (`c8241e0`) | **RESOLVED** |
| **P0-SWALLOW** | `core/`, `surgery/`, `trainer/` | Silent `except: pass` and bare `except:` swallowing exceptions and masking bugs | Replaced with specific exception handling and `# allowed-swallow` policy | Wave 2 (`d739210`) | **RESOLVED** |
| **P0-MOCK** | `benchmarks/suites/niah.py` | Fake NIAH passkey evaluation using `torch.randn` | Integrated with `neuralops.evaluation.niah_passkey` with fail-closed semantics | Wave 4 (`W4-U1`) | **RESOLVED** |
| **P0-IMPORT** | `neuralops/benchmarks/registry.py` | Import-time CUDA initialization and filesystem side effects | Implemented lazy initialization via `get_registry()` | Wave 1 (`SA-B1`) | **RESOLVED** |
| **P0-COLLIDE** | `tests/**` | Module name collisions in pytest across duplicate file basenames in subdirectories | Configured `--import-mode=importlib` in `pyproject.toml` | Wave 4 (`W4-R1`) | **RESOLVED** |
| **P0-ISOLATE** | `benchmarks/executor.py` | Subprocess crash or OOM bringing down master sweep runner | Wrapped worker execution in spawn context with graceful `EOFError`/`SIGKILL` traps | Wave 1 (`AGT-58`) | **RESOLVED** |

---

## 2. Primitives & Kernel Correctness Fixes (Wave 1 & 2)

| Bug ID | Subsystem / Component | Issue Description | Fix Applied | Remediation Scope | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FIX-01** | `ops/cuda/*` bindings | Missing `PYBIND11_MODULE` in 5 C++ extensions | Added explicit pybind11 module forward bindings | Wave 2 (`551e9d1495`) | **RESOLVED** |
| **FIX-02** | `ops/cuda/aegis_gate_norm` | NaN generation on near-zero norm inputs | Added numerical epsilon guard `fmaxf(norm, 1e-6f)` | Wave 2 (`dc3949e4d3`) | **RESOLVED** |
| **FIX-03** | `ops/cuda/favor_plus` | Precision underflow in random feature projections | Switched intermediate accumulator to FP32 | Wave 2 (`621bcf3817`) | **RESOLVED** |
| **FIX-04** | `ops/cuda/gla_unnormalized_scan` | Memory alignment violation on non-power-of-2 sequence lengths | Added grid boundary masking for arbitrary sequence lengths | Wave 2 (`875dbe8cd1`) | **RESOLVED** |
| **FIX-05** | `ops/triton/selective_scan` | Triton compilation error on odd channel dimensions | Parameterized tile blocking to handle non-power-of-2 dimensions | Wave 2 (`e5f2ed8ccf`) | **RESOLVED** |
| **FIX-06** | `layers/sliding_window.py` | Future-token causality leak on padded sequences | Enforced strict lower-triangular causal band masking | Wave 4 (`W4-L2`) | **RESOLVED** |
| **FIX-07** | `layers/gla.py` | Crash when optional bias parameter is `None` | Added conditional bias check `if self.bias is not None` | Wave 4 (`W4-L1`) | **RESOLVED** |
| **FIX-08** | `compression/hadamard_svd.py` | SVD reconstruction failure when matrix dimension is odd | Added padding to nearest even dimension before Walsh-Hadamard transform | Wave 2 (`f3841a0`) | **RESOLVED** |

---

## 3. Benchmarks & CLI Decomposition Fixes (Wave 1 & Wave 4)

| Bug ID | Subsystem / Component | Issue Description | Fix Applied | Remediation Scope | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CLI-01** | `benchmarks/cli.py` | Monolithic 2,922-line file violating single responsibility and maintainability | Decomposed into modular `cli_v2` subcommands (`benchmark`, `plan`, `results`, `doctor`, `ui`) | Wave 4 (`W4-I1`) | **RESOLVED** |
| **CLI-02** | `benchmarks/legacy_root/` | 58 unmaintained scripts polluting library root | Quarantined to `neuralops/benchmarks/_legacy/` with deprecation warnings | Wave 4 (`W4-I1`) | **RESOLVED** |
| **CLI-03** | `benchmarks/results/saver.py` | Concurrency race condition during parallel results write | Added atomic multi-file ledger writer with SQLite & Parquet sync | Wave 1 (`SA-B3`) | **RESOLVED** |
| **CLI-04** | `benchmarks/matrix.py` | L2 cache timing pollution during GPU benchmark sweeps | Added dynamic L2 cache eviction buffer based on device properties | Wave 1 (`SA-B4`) | **RESOLVED** |
| **CLI-05** | `pyproject.toml` | Console scripts referencing monolithic legacy `cli.py` | Migrated `neuralops` and `nops` entry points to `cli_v2.main:main` | Wave 4 (`W4-R1`) | **RESOLVED** |

---

## 4. Test Infrastructure & CI/CD Remediations (Wave 4)

| Bug ID | Subsystem / Component | Issue Description | Fix Applied | Remediation Scope | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **INF-01** | `tests/` | Flat disorganized test directory with over 440 loose test files | Structured test suite into 4 canonical tiers (`unit/`, `contract/`, `integration/`, `hardware/`) | Wave 4 (`W4-R1`) | **RESOLVED** |
| **INF-02** | `tests/conftest.py` | Incomplete shared fixtures causing duplicated setup logic | Implemented `tiny_bert`, `tiny_llama`, `mock_cuda_device`, `mock_hf_api`, `isolated_temp_ledger` | Wave 4 (`W4-R1`) | **RESOLVED** |
| **INF-03** | `.github/workflows/` | Outdated CI referencing deleted layout check scripts | Built matrix CI workflow for Python 3.10–3.12 with ruff, mypy, and contract tests | Wave 4 (`W4-R1`) | **RESOLVED** |
| **INF-04** | `tests/test_skip_ratio_check.py` | Unhandled `ModuleNotFoundError` during test collection | Added graceful `pytest.importorskip` guards for removed scripts | Wave 4 (`W4-R1`) | **RESOLVED** |
| **INF-05** | `examples/` | Missing runnable canonical examples for release verification | Added `quickstart.py`, `surgery_demo.py`, `compression_demo.py`, `hub_loading.py`, `serving_demo.py` | Wave 4 (`W4-R1`) | **RESOLVED** |
