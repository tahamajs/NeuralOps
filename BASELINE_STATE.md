# Repository Baseline State

Baseline captured during the repository-wide structure audit on 2026-11-03.

## Scope

- Repository: NeuralOps
- Working tree: `/Users/tahamajs/dev/Project`
- Tracked files at baseline: 13,318
- Visible non-bytecode worktree files from `rg --files`: 2,961
- Local environment directories and caches are excluded from production source decisions.

## Preservation rules

- Existing user modifications are preserved.
- No source, compatibility module, benchmark, or test has been deleted or moved during the baseline pass.
- New audit documents are additive and do not replace existing policy documents.
- Destructive cleanup requires a caller/export/registry/test/documentation evidence record first.

## Current audit state

- Full filesystem inventory: captured.
- Package and test directory map: captured.
- Entry-point and registry locations: identified.
- Cross-file caller graph: in progress.
- Duplicate and dead-file decisions: pending dependency evidence.
- Canonical CUDA documentation reconciliation: 110 catalog modules and 114 native implementations; historical count references are explicitly labeled.
