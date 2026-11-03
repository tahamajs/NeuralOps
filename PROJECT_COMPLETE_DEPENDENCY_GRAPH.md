# Complete Dependency Graph Audit

This document is the working record for cross-file relationships. It is intentionally evidence-based; a missing edge means “not yet checked,” not “no dependency.”

## Graph layers

```text
User / CI / PBS command
  -> CLI, script, notebook, or test entry point
    -> runner / benchmark adapter / public API
      -> registry, factory, or dispatcher
        -> canonical implementation or compatibility wrapper
          -> backend operation (CUDA / Triton / PyTorch)
            -> result writer, report, or assertion
```

## Confirmed anchors

- Package metadata and entry points: `pyproject.toml`.
- Registry definitions and accessors: `neuralops/core/registry.py`.
- Static catalog: `neuralops/benchmarks/static_catalog.json`.
- Public kernel catalogs: `neuralops/kernels/cuda/` and `neuralops/kernels/triton/`.
- Core kernel catalogs: `neuralops/core/cuda_kernels/` and `neuralops/core/triton_kernels/`.
- Standalone benchmark launchers: `scripts/benchmarks/`.
- Importable benchmark components: `neuralops/benchmarks/`.
- GPU matrix tests: `tests/gpu_exhaustive/`.
- Remote execution policy: `AGENTS.md` and `.agents/skills/hpc-sharif-cluster/SKILL.md`.
- Confirmed symbol callers: `PROJECT_CALLER_MAP.md`.

## Required edge checks

For each production module, the audit must check imports, exports, registry decorators, dynamic import strings, CLI references, benchmark references, test references, notebook references, and documentation links before assigning a cleanup decision.

## Status

The high-level graph is documented. Confirmed cross-cutting edges include the lazy import layer (`neuralops/__init__.py` and `neuralops/_lazy.py`), registry consumers under `neuralops/benchmarks`, `tests`, and `scripts`, and the public CUDA/Triton catalog APIs under `neuralops/kernels`. Symbol-level and file-level edges are still being enumerated; no file is classified as dead until that enumeration is complete.

## Evidence snapshot

- Tracked files: 13,318 (`git ls-files`).
- Visible worktree files: 2,961 (`rg --files`).
- Python files: 2,220 tracked files by extension inventory.
- Package markers: 378 tracked `__init__.py` files.
- Test subsystem directories: 24.
- Canonical registries: 131 attention keys and 72 compressor keys, as asserted by the repository registry tests.
- Canonical CUDA catalog: 110 modules and 114 native implementations; raw source/support directory counts are not used as registry counts.
- Canonical Triton catalog: 26 modules and 62 unified operators.
- Notebooks: 29.
- Compatibility/dynamic-import evidence: 630 tracked Python files reference at least one
  legacy/public kernel or package namespace; this is why similarly named trees cannot be
  removed based on filename similarity alone.
