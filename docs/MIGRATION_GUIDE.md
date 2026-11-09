# NeuralOps Migration Guide

This guide details breaking changes, deprecated constructs, migration steps, and the formal API deprecation policy across releases of NeuralOps.

---

<a id="deprecation-policy"></a>
## Deprecation Policy & Semantic Versioning Contract

NeuralOps follows [Semantic Versioning 2.0.0](https://semver.org/) (`MAJOR.MINOR.PATCH`) with strict deprecation lifecycle guarantees (Roadmap §5.1 & §5.5):

1. **Module & Class Renaming:**
   - Any renamed module, class, or public symbol remains available at its legacy location for at least **one full MINOR release cycle**.
   - Importing or instantiating a renamed construct raises an explicit `DeprecationWarning` detailing the replacement symbol (never silent, F-82).
   - All renames must be documented in both `docs/MIGRATION_GUIDE.md` and `docs/RENAME_MAP.md` before final symbol removal is permitted in a subsequent MAJOR or MINOR release.

2. **Alias Lifecycle & Removal:**
   - Backwards-compatibility aliases (e.g. registry alias strings or deprecated function names) are retained for at least one MINOR release.
   - Removal occurs only in a MINOR or MAJOR release and must be documented in `CHANGELOG.md` under `Breaking Changes`.

3. **Public Function Signature Evolution:**
   - When parameter names or signatures change, the legacy parameter name is retained as an optional argument emitting a `DeprecationWarning` rather than being silently dropped or causing immediate breakage.

4. **Security Defaults as Breaking Changes:**
   - Any hardening change to security defaults (such as switching `trust_remote_code` default to `False` or binding servers strictly to `127.0.0.1`) is treated as a **BREAKING change**.
   - Such changes must be highlighted in `CHANGELOG.md` (Security & Breaking Changes), documented in `docs/MIGRATION_GUIDE.md`, and noted in `docs/TROUBLESHOOTING.md` (F-87, F-89).

5. **SemVer Evolution Rules:**
   - **PATCH (`x.y.Z`):** Backward-compatible bug fixes and internal performance enhancements only. Zero public API changes.
   - **MINOR (`x.Y.z`):** New features, new kernels/models, non-breaking additions, and planned deprecations. In the `v2.x` release series, breaking changes accompanying significant architectural modernizations are permitted in MINOR releases provided they are accompanied by explicit migration notes, deprecation shims, and changelog warnings.
   - **MAJOR (`X.y.z`):** Broad architectural redesigns, end-of-life symbol removals, and incompatible contract changes.

---

## Migrating to Version 2.0.x / 2.1.0

### 1. Canonical Registry Naming & Alias Unification
Prior releases permitted ambiguous string aliases to be registered as separate canonical keys. In v2.0+, each unique implementation class has exactly one canonical key:
- **Attention mechanisms**: `svd_linear_attn` and `svd_linear_inject` are aliases for canonical `svd_linear`.
- **Compressors**: `asvd_compressor` is an alias for canonical `asvd`. `hadamard_svd` is an alias for canonical `hadamard_incoherent_svd`.
- **Duplicate Registration**: Attempting to register two classes under the same canonical key raises a `RegistryError`.

### 2. CLI Invocation: Web UI vs Terminal UI
The CLI command `neuralops ui` now launches the modern FastAPI + htmx Web UI dashboard:
- **Web UI**: Run `neuralops ui` (flags: `--port`, `--host`, `--share`, `--auth`).
- **Terminal UI**: The interactive two-pane UNIX terminal inspector is now accessed via `neuralops tui`:
  ```bash
  # Old (v1.x)
  neuralops ui --snapshot

  # New (v2.0+)
  neuralops tui --snapshot
  ```

### 3. Version Source of Truth & Configuration Consolidation
- Standalone `ruff.toml` and `pytest.ini` have been consolidated into `pyproject.toml`.
- Package version is authoritative in `pyproject.toml` and synchronized via `python scripts/utils/sync_version.py --check`.

### 4. Direct PyTorch References Centralization
PyTorch fallback reference implementations are now centrally exported from `neuralops.ops.pytorch`. Direct imports from individual un-exported test folders should be redirected:
```python
# Recommended
from neuralops.ops.pytorch import gdn_scan_pytorch, fused_rmsnorm_pytorch
```

### 5. CLI Entrypoint Consolidation & Deprecated Shims (AGT-54, Decision 3)
In prior releases, three parallel CLI parsers existed (`neuralops.benchmarks.cli`, `neuralops.cli.main`, and `neuralops.cli.core.app`).
- **Canonical Entrypoint**: All CLI invocations, entry points, and console scripts now route solely to `neuralops.benchmarks.cli:main`.
- **Deprecated Shims**: `neuralops.cli.main` and `neuralops.cli.core.app` are deprecated backward-compatibility shims that emit `DeprecationWarning` and delegate directly to `neuralops.benchmarks.cli:main`. They will be removed in v3.0.0.
