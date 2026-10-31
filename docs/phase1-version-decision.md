## Phase 1 Version Decision

**Date:** 2026-10-31
**Owner approval:** Yes
**Decision:** Phase 1 operates at `__version__ = "1.9.0"` (in development) on top of the Phase 0 spec commit `21ebdeaa63`.

**Rationale:**
- Phase 1 delivers NeuralOps 1.9.0 (minor version bump from 1.8.1)
- The NRL spec (`docs/recipe-language-spec.md`) is the first 1.9.0 artifact
- Version is bumped at start of Phase 1, not at end, to avoid drift
- Final `v1.9.0` release tag will NOT be created in Phase 1 — only `v1.9.0-phase1` milestone tag

**What changed vs Phase 1 prompt's default:**
- Invariant #9 relaxed from `main at v1.8.1` to `main at 21ebdeaa63 or later`
- All other invariants unchanged
