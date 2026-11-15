# Wave 6 Hotfix Backlog

This backlog tracks findings, edge cases, minor doc mismatches, and improvements discovered during Wave 5 integration, verification, and acceptance testing.
Per Wave 5 protocol, findings are logged here rather than patched inline to prevent regressions.

| Finding ID | Discovered By | Subsystem | Severity | File / Symbol | Description | Recommended Fix |
|---|---|---|---|---|---|---|
| W5-12-01 | W5-12 | docs | LOW | `README.md` | README claims "131 Attention Mechanisms" while physical canonical registry has 118 native classes (expands to 131 with external FLA extension keys and aliases). | Clarify taxonomy in README: 118 canonical internal classes + external FLA keys total 131. |
| W5-12-02 | W5-12 | docs | LOW | `README.md` | README claims "72 Compressors" while physical registry has expanded to 89 registered keys (73 distinct classes). | Update README marketing copy to reflect 89 registered compression engines (73 distinct classes). |
| W5-12-03 | W5-12 | api | LOW | `neuralops/api.py` | `replace_attention` and `compress` had undocumented `validation_level` parameter; `certify` was missing from `api.py`. | Fixed inline in W5-12: updated docstrings and exposed `certify` in `api.py` and `__init__.py`. |
| W5-10-01 | W5-10 | benchmarks | Medium (Hygiene) | `neuralops/benchmarks/**/*.py` | 456 naked `print()` calls across benchmark scripts (terminal_ui.py, _legacy/*.py, etc.). | Replace naked prints with `logger.info(...)` or delegate output to `cli_v2` Rich console. |
| W5-10-02 | W5-10 | evaluation | Medium (Hygiene) | `neuralops/evaluation/**/*.py` | 57 naked `print()` calls across evaluation scripts (full_mnli_validation.py, etc.). | Replace prints with `logger.info(...)` using module-level logger. |
| W5-10-03 | W5-10 | runners | Medium (Hygiene) | `neuralops/runners/**/*.py` | 22 naked `print()` calls in orchestration scripts (unified_orchestrator.py, etc.). | Replace prints with structured runner logging. |
| W5-10-04 | W5-10 | analysis | Low (Hygiene) | `neuralops/analysis/**/*.py` | 14 naked `print()` calls in module analysis routines (main.py, etc.). | Replace with `logger.info(...)`. |
| W5-10-05 | W5-10 | utils | Low (Hygiene) | `neuralops/utils/**/*.py` | 10 naked `print()` calls in utility profilers (benchmark_profiler.py, etc.). | Replace with `logger.debug(...)` / `logger.info(...)`. |
| W5-10-06 | W5-10 | cross-cutting logging | Low (Performance) | `neuralops/**/*.py` | 301 eager f-string allocations inside `logger.*(f"...")` calls. | Convert eager f-strings to lazy `%s` arguments to eliminate hot-path string allocation overhead. |
| W5-10-07 | W5-10 | hub / docs | Low (Tooling) | `neuralops/hub/*.py` | 5 string literals matching `print(` inside generated markdown model cards. | False positives from naive scanners; harmless, but annotate or keep documented. |
| W5-18-01 | W5-18 | evaluation / devices | Medium (Portability) | `neuralops/evaluation/standalone/*.py` | 11 standalone scripts default to `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` rather than parameterizing device. | Standardize device resolution via `neuralops.cli.core.devices.resolve_device` or CLI `--device`. |
| W5-18-02 | W5-18 | benchmarks / kd | Low (Tech Debt) | `neuralops/benchmarks/kd/teacher_cache.py:66` | Legacy PyTorch <2.0 `weights_only` TypeError fallback remains in codebase. | Deprecate PyTorch <2.0 fallback since NeuralOps v2.4+ requires PyTorch >= 2.0. |
| W5-18-03 | W5-18 | security / ci | Low (Tooling) | CI workflows | `pip-audit` missing from local environment. | Add automated `pip-audit` step to CI security audit matrix (`.github/workflows/security.yml`). |
