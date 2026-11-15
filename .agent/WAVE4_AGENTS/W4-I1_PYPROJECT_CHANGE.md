# Coordination: Entry Point Update for pyproject.toml

**From:** Agent W4-I1 (Benchmarks CLI Decomposition & Legacy Quarantine)  
**To:** Agent W4-R1 (Infrastructure + Release)  
**Subject:** Update `[project.scripts]` entrypoints to CLI v2

---

### Context

Agent W4-I1 has completed the decomposition of the monolithic `neuralops/benchmarks/cli.py` (2,922 lines) into `neuralops/benchmarks/cli_v2/` with:
- Top-level canonical entrypoint: `neuralops.benchmarks.cli_v2.main:main`
- Modular subcommands under `neuralops/benchmarks/cli_v2/commands/` (`benchmark`, `plan`, `results`, `doctor`, `ui`, `completion`, plus `run` alias)
- Rendering and completion utilities under `neuralops/benchmarks/cli_v2/utils/`
- Full backward compatibility delegation: legacy subcommands and `neuralops.benchmarks.cli.main` deprecation shim
- 58 legacy scripts quarantined under `neuralops/benchmarks/_legacy/` with deprecation warnings

### Requested Change in `pyproject.toml`

Please update `[project.scripts]` in `pyproject.toml` to point `neuralops`, `neuralops-cli`, `nops`, and `nops-cli` directly to the new canonical CLI v2 entrypoint:

```toml
[project.scripts]
neuralops = "neuralops.benchmarks.cli_v2.main:main"
neuralops-cli = "neuralops.benchmarks.cli_v2.main:main"
nops = "neuralops.benchmarks.cli_v2.main:main"
nops-cli = "neuralops.benchmarks.cli_v2.main:main"
neuralops-eval = "neuralops.evaluation.standalone.cli:main"
neuralops-run = "neuralops.main:main"
```

Also, if package inclusion patterns in `[tool.setuptools.packages.find]` or lint rules in `[tool.ruff.lint.per-file-ignores]` mention `neuralops/benchmarks/legacy_root` or `legacy_scripts`, add `neuralops/benchmarks/_legacy/**` as appropriate.
