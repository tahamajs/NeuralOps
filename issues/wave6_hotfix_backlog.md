# Wave 6 Hotfix Backlog

This backlog tracks findings, edge cases, minor doc mismatches, and improvements discovered during Wave 5 integration, verification, and acceptance testing.
Per Wave 5 protocol, findings are logged here rather than patched inline to prevent regressions.

| Finding ID | Discovered By | Subsystem | Severity | File / Symbol | Description | Recommended Fix |
|---|---|---|---|---|---|---|

## W5-11: mypy Typing Violations (2026-11-15T11:41:43Z)

| Scope | Error Count |
|---|---|
| mypy --strict (core modules) | 948 |
| mypy full repo (--ignore-missing-imports) | 897 |

### Top Error Categories (strict mode):
- [arg-type]: 252 occurrences
- [operator]: 123 occurrences
- [assignment]: 119 occurrences
- [no-any-return]: 88 occurrences
- [index]: 79 occurrences
- [union-attr]: 76 occurrences
- [no-untyped-def]: 37 occurrences
- [type-arg]: 34 occurrences
- [no-untyped-call]: 33 occurrences
- [return-value]: 17 occurrences

**Remediation:** These must be fixed before v2.5.0 strict typing gate.
| W5-18-01 | W5-18 | evaluation / devices | Medium (Portability) | `neuralops/evaluation/standalone/*.py` | 11 standalone scripts default to `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")` rather than parameterizing device. | Standardize device resolution via `neuralops.cli.core.devices.resolve_device` or CLI `--device`. |
| W5-18-02 | W5-18 | benchmarks / kd | Low (Tech Debt) | `neuralops/benchmarks/kd/teacher_cache.py:66` | Legacy PyTorch <2.0 `weights_only` TypeError fallback remains in codebase. | Deprecate PyTorch <2.0 fallback since NeuralOps v2.4+ requires PyTorch >= 2.0. |
| W5-18-03 | W5-18 | security / ci | Low (Tooling) | CI workflows | `pip-audit` missing from local environment. | Add automated `pip-audit` step to CI security audit matrix (`.github/workflows/security.yml`). |
