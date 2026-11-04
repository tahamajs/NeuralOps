# Project Cleanup Candidates

This file is an evidence ledger, not a deletion plan. No candidate may be removed until its import, export, registry, test, CLI, notebook, configuration, and documentation references have been checked.

## Initial candidates requiring cross-reference

| Area | Why it needs review | Required evidence before action |
|---|---|---|
| `neuralops/attentions/` vs `neuralops/layers/` | The repository documents `layers/` as canonical and describes `attentions/` as a compatibility surface. | Complete import/export and registry map; preserve public aliases. |
| `neuralops/compression_engine/` vs `neuralops/compression/` | Both contain family-shaped implementations. | Identify canonical implementations, compatibility imports, and duplicate symbols. |
| `neuralops/kernels/` vs `neuralops/core/*_kernels/` | Public catalog wrappers and core catalogs expose overlapping names. | Compare exports and `list_*_kernels()` behavior; do not remove wrappers. |
| `scripts/benchmarks/` vs `neuralops/benchmarks/` | Standalone launchers and importable benchmark components share concepts. | Map every command, test, and entry point. |
| `docs/results/`, `results/`, and `academic/` | Multiple result/documentation trees may contain generated or archival copies. | Check `.gitignore`, provenance, and links before archive or removal. |
| `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `dist/`, `*.egg-info/` | Generated local outputs are present in the checkout. | Confirm ignore policy and remove only with explicit cleanup scope. |

## Current policy

- No deletion has been performed.
- No public compatibility surface is assumed dead from directory naming alone.
- Historical documents remain until their references and archival role are established.
- Generated artifacts are not moved into production source directories.
