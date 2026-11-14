# Breaking Changes & Deprecations Report for Wave 4 Release (W4-R1)

**From:** Agent W4-I1  
**To:** Agent W4-R1 (Infrastructure + Release)  
**Date:** 2026-11-15  

---

### 1. CLI Entrypoint Modernization

- **Old entrypoint:** `neuralops.benchmarks.cli:main`
- **New canonical entrypoint:** `neuralops.benchmarks.cli_v2.main:main`
- **Deprecation Shim:** `neuralops/benchmarks/cli.py` is maintained as a deprecation shim emitting `DeprecationWarning` on invocation and delegating to CLI v2.
- **Decomposed Subcommands:**
  - `neuralops benchmark` (`neuralops.benchmarks.cli_v2.commands.benchmark`)
  - `neuralops run` (canonical alias for benchmark)
  - `neuralops plan` (`neuralops.benchmarks.cli_v2.commands.plan`)
  - `neuralops results` (`neuralops.benchmarks.cli_v2.commands.results`)
  - `neuralops doctor` (`neuralops.benchmarks.cli_v2.commands.doctor`)
  - `neuralops ui` (`neuralops.benchmarks.cli_v2.commands.ui`)
  - `neuralops completion` (`neuralops.benchmarks.cli_v2.utils.completion`)
- **Backward Compatibility:** Legacy subcommands (`multiseed`, `matrix`, `compare`, `list`, `info`, `resume`, `verify`, `export`, etc.) are gracefully routed with `DeprecationWarning`.

### 2. Legacy Benchmark Script Quarantine

- **Quarantine Location:** `neuralops/benchmarks/_legacy/`
- **Quarantined Scripts:** 58 legacy scripts across `legacy_scripts/` (31 scripts), `legacy_scripts/legacy/` (10 scripts), and `legacy_root/` (19 scripts) have been migrated into `_legacy/`.
- **Deprecation Policy:** Every script in `_legacy/` issues a `DeprecationWarning` on module import/execution. Forwarding shims remain in `legacy_scripts/` and `legacy_root/` to prevent immediate breakage for legacy callers and tests during the v2.4 deprecation cycle, slated for removal in v2.5.
