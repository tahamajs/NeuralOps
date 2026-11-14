# NeuralOps Wave 3 — 20-Agent Comprehensive Completion

**Version:** 1.0
**Date:** 2026-11-14
**Scope:** Everything NOT covered by Wave 1 (5 agents) or Wave 2 (20 agents)
**Prerequisite:** Wave 1 AND Wave 2 must be merged and green
**Duration:** 6-10 weeks
**Coordinator:** Human reviewer

---

## 0. Preamble — What This Wave Covers

Wave 1 fixed 45 known P0/P1 bugs across 5 subsystems.
Wave 2 deep-audited ~60 files, sampled ~100 files, fixed ~30 more P0s.

**Wave 3 closes the remaining gaps:**

1. **Deep-audit completion:** 91 layers + 37 compressors + 60 ops
2. **Infrastructure:** tests, CI/CD, pre-commit, coverage gates
3. **Documentation:** docs/, README, migration guides
4. **Packaging:** pyproject.toml, configs, entry points
5. **Benchmarks cleanup:** cli.py decomposition, 58 legacy scripts, P1/P2 remaining
6. **Security:** torch.load, secret scanning, preprocess sandbox
7. **Cross-cutting quality:** error hierarchy, logging, type hints
8. **Real-world integration tests:** HF Hub, all 6 zoos, all 131 attentions, all 72 compressors
9. **Release:** CHANGELOG, BREAKING_CHANGES, verification, tagging

**After Wave 3:** the library is production-ready for v2.3.0 release.

---

## 1. Definition of Done (v2.3.0 Release Criteria)

### 1.1 — Coverage
- [ ] **All 131 attention mechanisms** deep-audited (Wave 2 covered 40, Wave 3 covers 91)
- [ ] **All 72 compressors** deep-audited (Wave 2 covered 35, Wave 3 covers 37)
- [ ] **All ~100 Triton/CUDA kernels** deep-audited (Wave 2 covered 40, Wave 3 covers 60)
- [ ] **All 6 HF zoo types** load end-to-end
- [ ] **All 26 Triton flagship kernels** run correctly
- [ ] Every file has an audit verdict in `audits/wave3_master.md`

### 1.2 — Infrastructure
- [ ] Test directory reorganized: `tests/{unit,contract,integration,hardware}/`
- [ ] Pytest markers defined: `@pytest.mark.{fast,gpu,slow,hub}`
- [ ] Shared fixtures in `tests/conftest.py`
- [ ] `.github/workflows/ci.yml` runs on PR with matrix (py3.10, 3.11, 3.12)
- [ ] `.github/workflows/gpu.yml` runs on GPU runner
- [ ] `.pre-commit-config.yaml` with ruff, mypy, license header
- [ ] Coverage gate: 70% line coverage (documented exceptions)

### 1.3 — Documentation
- [ ] `docs/` all files audit-verified
- [ ] `README.md` claims verified against code
- [ ] `docs/migrations/v2.2_to_v2.3.md` written
- [ ] All public API docstrings verified
- [ ] 5+ runnable examples in `examples/`
- [ ] `docs/benchmarks/` exists and matches actual benchmarks

### 1.4 — Packaging
- [ ] `pyproject.toml` audited: entry points, extras, deps
- [ ] `configs/*.yaml` all valid (schema-validated)
- [ ] `python -m build` succeeds
- [ ] `pip install dist/*.whl` succeeds
- [ ] All 3 entry points (`neuralops`, `nops`, `neuralops-cli`) resolve

### 1.5 — Benchmarks Cleanup
- [ ] `benchmarks/cli.py` decomposed into `benchmarks/cli_v2/` (Phase 6)
- [ ] 58 legacy scripts migrated or quarantined (Phase 7)
- [ ] Remaining P1/P2 in `benchmarks/{schema,saver,summarizer,reproduce,index,hf_sync,git_sync,dedup,dispatcher}.py` fixed

### 1.6 — Security
- [ ] `torch.load(weights_only=True)` everywhere (except documented)
- [ ] `scan_for_secrets` patterns robust (5+ token types)
- [ ] `load_preprocess_script` requires `trust=True` opt-in
- [ ] `hf_token` never logged or serialized
- [ ] `pip-audit` green

### 1.7 — Quality
- [ ] Error hierarchy unified (all `NeuralOpsError` subclasses)
- [ ] Logging consistent (`logging.getLogger(__name__)` everywhere, no `print`)
- [ ] Type hints on all public API
- [ ] Dead code removed
- [ ] Import organization (stdlib → 3rd-party → neuralops)

### 1.8 — Release
- [ ] `CHANGELOG.md` v2.3.0 entry complete
- [ ] `BREAKING_CHANGES.md` complete
- [ ] `AUDIT_MASTER_FIXES.md` complete
- [ ] `VERIFICATION.md` complete
- [ ] `git tag v2.3.0 && git push --tags`
- [ ] 3 reviewer sign-offs

---

## 2. The Iron Law — File Ownership (20 Agents)

**Each file has exactly ONE owner. NO two agents touch the same file.**

### 2.1 — File Ownership Map

| Agent | Domain | Owned Paths | Excluded Paths |
|---|---|---|---|
| **FA** | Layers deep-audit (linear) | 30 specific files in `neuralops/layers/` | other layers |
| **FB** | Layers deep-audit (sparse) | 30 specific files in `neuralops/layers/` | other layers |
| **FC** | Layers deep-audit (standard) | 30 specific files in `neuralops/layers/` | other layers |
| **FD** | Layers deep-audit (specialized) | 30 specific files in `neuralops/layers/` | other layers |
| **OA** | Ops deep-audit (Triton) | 50 files in `neuralops/ops/triton/**` | other ops |
| **OB** | Ops deep-audit (CUDA + rest) | 50 files in `neuralops/ops/cuda/**` + `neuralops/ops/*.py` | other ops |
| **CA** | Compression deep-audit | 37 specific files in `neuralops/compression/**` | — |
| **TA** | Test infrastructure | `tests/**` (organization, markers, fixtures) + `pyproject.toml` (pytest config only) | `tests/hub/**` |
| **TB** | CI/CD + Pre-commit | `.github/**`, `.pre-commit-config.yaml`, `Makefile`, `scripts/ci_*.sh` | — |
| **DA** | Documentation audit | `docs/**`, `README.md` | `docs/migrations/**` |
| **DB** | Migration guides + examples | `docs/migrations/**`, `examples/**`, `notebooks/**` | other docs |
| **PA** | Packaging + Configs | `pyproject.toml`, `setup.py`, `setup.cfg`, `configs/**`, `MANIFEST.in` | `pyproject.toml` pytest section |
| **BA** | Benchmarks cli decomposition | `benchmarks/cli_v2/**` + `benchmarks/cli.py` | other benchmarks |
| **BB** | Benchmarks legacy migration | `benchmarks/_legacy/**` + `benchmarks/legacy_root/**` + `benchmarks/legacy_scripts/**` | other benchmarks |
| **BC** | Benchmarks P1/P2 remaining | `benchmarks/schema.py`, `saver.py`, `summarizer.py`, `reproduce.py`, `index.py`, `hf_sync.py`, `git_sync.py`, `dedup.py`, `dispatcher.py`, `paths.py` | `benchmarks/cli.py`, legacy |
| **SA** | Security hardening | `neuralops/utils/hf_git_exporter.py`, `neuralops/hub/uploader.py`, `neuralops/core/exceptions.py` | — |
| **QA** | Cross-cutting quality | `neuralops/core/exceptions.py`, `neuralops/utils/logging_config.py` | — |
| **IA** | Real-world integration tests | `tests/hub/**` + `tests/integration/**` | `tests/unit/**`, `tests/contract/**`, `tests/hardware/**` |
| **RA** | Release consolidation | `CHANGELOG.md`, `BREAKING_CHANGES.md`, `AUDIT_MASTER_FIXES.md`, `VERIFICATION.md`, `_version.py` | — |
| **RB** | Final verification + tagging | `audits/wave3_master.md`, git tag | — |
