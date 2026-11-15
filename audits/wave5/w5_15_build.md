# Wave 5 Release Engineering Audit Report: W5-15
# Version Bump, PEP 517 Build & CLI Entrypoint Verification

**Agent ID:** `W5-15`  
**Phase:** 4 — Release Engineering  
**Branch:** `wave5/W5-15`  
**Target Version:** `2.4.0`  
**Audit Date:** 2026-11-15  
**Status:** `PASS` (100% Verified)

---

## 1. Executive Summary

Wave 5 subagent `W5-15` executed an end-to-end release engineering audit and packaging verification for the NeuralOps platform at canonical version `2.4.0`.

The audit verified:
1. **Version Metadata Synchronization**: `pyproject.toml` and `neuralops/_version.py` define version `2.4.0`. Downstream metadata across `pyproject.cuda.toml`, `CITATION.cff`, `SECURITY.md`, `README.md`, `neuralops/README.md`, `examples/README.md`, and `docs/api/*.md` were synchronized and validated via `scripts/utils/sync_version.py --check`.
2. **Standard PEP 517 Build**: Executed `python -m build` generating compliant source distribution (`sdist`) and pure-Python wheel (`bdist_wheel`) in `dist/`. Zero build errors encountered.
3. **Submodule Discovery Integrity**: Wheel contents verified via `unzip -l`, confirming inclusion of all canonical subpackages: `neuralops/layers/linear/`, `neuralops/benchmarks/cli_v2/`, `neuralops/surgery/director.py`, `neuralops/compression/`, `neuralops/attentions/`, and runtime engines.
4. **Isolated Sandbox Installation**: Created clean temporary virtual environment (`/tmp/neuralops_release_sandbox`), successfully installed `neuralops-2.4.0-py3-none-any.whl` and dependencies, and confirmed clean non-editable imports.
5. **CLI Entrypoint Verification**: Verified that all registered console scripts (`neuralops`, `nops`, `neuralops-cli`, `nops-cli`) execute cleanly with `--version` and emit `NeuralOps 2.4.0` with exit code 0.

---

## 2. Distribution Artifacts Metadata Table

| Artifact Filename | Distribution Type | File Size | SHA-256 Checksum |
|---|---|---|---|
| `neuralops-2.4.0-py3-none-any.whl` | Binary Wheel (`bdist_wheel`) | 4,830,147 bytes (4.6 MB) | `e69c9dfed5576bc7176c6852d090acfa93e5d7bbb2cc6af99eb6d5a945b64245` |
| `neuralops-2.4.0.tar.gz` | Source Distribution (`sdist`) | 3,591,488 bytes (3.4 MB) | `27ab55da65ef7d21fcbd7f5dc2aac8e17613c3374ed2db743ce2d008fd4d6ea0` |

---

## 3. Wheel Table of Contents Inspection Findings

Inspection of `neuralops-2.4.0-py3-none-any.whl` confirmed complete subpackage discovery and packaging:

| Target Subpackage / Critical Module | Status in Wheel | Archive Path Verified |
|---|---|---|
| CLI v2 Core | Present | `neuralops/benchmarks/cli_v2/main.py`, `neuralops/benchmarks/cli_v2/__init__.py` |
| CLI v2 Subcommands | Present | `neuralops/benchmarks/cli_v2/commands/{benchmark,doctor,plan,results,ui}.py` |
| Linear Attention Zoo | Present | `neuralops/layers/linear/{gla_attention,retnet_attention,cosformer,spectra_v2,...}.py` |
| Universal Surgery Director | Present | `neuralops/surgery/director.py` |
| Compression Engines | Present | `neuralops/compression/` & `neuralops/compression_engine/` |
| Attention Mechanisms | Present | `neuralops/attentions/` |
| Runtime & Serving | Present | `neuralops/serving/`, `neuralops/runtime/`, `neuralops/trainer/` |
| Dist-info Metadata | Present | `neuralops-2.4.0.dist-info/{METADATA,WHEEL,entry_points.txt,licenses/LICENSE}` |

---

## 4. Isolated Sandbox Installation & CLI Entrypoint Verification

An isolated, non-editable virtual environment was provisioned at `/tmp/neuralops_release_sandbox`. The wheel was installed using standard `pip install dist/neuralops-2.4.0-py3-none-any.whl`.

### Entrypoint Test Results

| Entrypoint Binary | Invocation | Observed Output | Exit Code | Evaluation |
|---|---|---|---|---|
| `neuralops` | `neuralops --version` | `NeuralOps 2.4.0` | 0 | **SUCCESS** |
| `nops` | `nops --version` | `NeuralOps 2.4.0` | 0 | **SUCCESS** |
| `neuralops-cli` | `neuralops-cli --version` | `NeuralOps 2.4.0` | 0 | **SUCCESS** |
| `nops-cli` | `nops-cli --version` | `NeuralOps 2.4.0` | 0 | **SUCCESS** |

Interactive `--help` test executed cleanly, rendering the CLI v2 Rich UI banner and registering all subcommands (`surgery`, `compose`, `recommend`, `calibrate`, `compress`, `search`, `export`, `serve`, `generate`, `niah`, `benchmark-fused`, `distill`, `benchmark`, `plan`, `results`, `resume`, `monitor`, `doctor`, `info`, `inspect`, `roofline`, `validate`, `scaffold`, `list`, `verify`, `ui`).

---

## 5. Remediated Related Files

In accordance with the Holistic Fix Mandate (Spec Section 1.3):
1. **`neuralops/__init__.py`**: Added a scoped warning filter (`_warnings.catch_warnings()`) around the `pkg_resources` compatibility shim to prevent setuptools deprecation notices from polluting CLI stdout/stderr.
2. **`scripts/utils/sync_version.py`**: Executed synchronization of version `2.4.0` across downstream metadata files (`pyproject.cuda.toml`, `CITATION.cff`, `SECURITY.md`, `README.md`, `neuralops/README.md`, `examples/README.md`, `docs/api/*.md`).
3. **`tests/test_version_sync.py`**: Updated hardcoded legacy `2.1.0` references in test assertions to canonical version `2.4.0` (`2.4.x` active security policy), ensuring test suite pass rate is 100% (4/4 passed).

---

## 6. Archived Audit Evidence

All execution transcripts, build logs, and entrypoint outputs are archived under `audits/wave5/w5_15_evidence/`:
- `version_check.txt`: Programmatic import check output confirming `neuralops.__version__ == "2.4.0"`.
- `build_output.txt`: PEP 517 build standard output and error log.
- `build.log`: Complete log of sdist and wheel generation.
- `dist_contents.txt`: Detailed directory listing and sizes of `dist/`.
- `wheel_filelist.txt`: `unzip -l` archive listing of the built wheel.
- `wheel_contents.txt`: Duplicate copy ensuring compatibility with multi-agent naming conventions.
- `sandbox_install.log`: Standard output of pip installation inside isolated sandbox environment.
- `entrypoint_verification.txt`: Subprocess execution trace of all CLI entrypoints.
- `cli_verification.txt`: Duplicate copy ensuring compatibility with multi-agent naming conventions.

---

## 7. Acceptance Criteria Checklist

- [x] Version updated to `2.4.0` in `neuralops/_version.py` and `pyproject.toml`.
- [x] `neuralops --version` prints `NeuralOps 2.4.0` after `pip install "$WHL" --force-reinstall`.
- [x] Wheel builds cleanly with zero errors via PEP 517 `python -m build`.
- [x] Wheel installs in sandbox venv and CLI entrypoints return `NeuralOps 2.4.0`.
- [x] Evidence files archived in `audits/wave5/w5_15_evidence/` including `build_output.txt`.
- [x] Report `audits/wave5/w5_15_build.md` written and committed.
- [x] Progress updated in `.agent/WAVE5_PROGRESS.json`.
