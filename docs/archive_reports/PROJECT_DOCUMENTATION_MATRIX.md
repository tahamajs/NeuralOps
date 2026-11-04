# Project Documentation Matrix

| Topic | Canonical document | Supporting documents | Consistency requirement |
|---|---|---|---|
| Agent operation and verification | `AGENTS.md` | `.agents/skills/` | Agent rules must not contradict repository policy. |
| Contribution workflow | `CONTRIBUTING.md` | `SECURITY.md`, `CODE_OF_CONDUCT.md` | Contributor instructions remain authoritative. |
| User overview | `README.md` | `docs/guides/`, `examples/` | Commands and counts must resolve to current files. |
| Repository structure | `PROJECT_STRUCTURE.md` | `PROJECT_FILE_INVENTORY.md`, `PROJECT_DEPENDENCY_MAP.md`, `PROJECT_CALLER_MAP.md` | Paths must exist or be explicitly marked archival. |
| HPC execution | `HPC_GPU_MASTER_GUIDE.md` | `.agents/skills/hpc-sharif-cluster/` | PBS, environment, and host rules must agree with `AGENTS.md`. |
| Benchmark integrity | benchmark test files and benchmark scripts | `docs/benchmarks/` | Results must be real, fail-closed, and traceable. |
| Versioning | `AGENTS.md` §14 | `pyproject.toml`, `CHANGELOG.md`, `CITATION.cff` | Version source-of-truth rules must remain synchronized. |

## Review status

- Root policy documents identified.
- Structure and dependency guide added.
- Canonical CUDA claims were reconciled to 110 catalog modules and 114 native implementations; raw compatibility/support directories are now described separately.
- Remaining historical count references are explicitly labeled as historical snapshots or preserved notebook titles.
- Historical and generated documentation requires provenance review before consolidation.
- README release-gate language now separates historical GPU evidence from the current
  exhaustive gate and no longer claims that `scripts/benchmarks` was deleted.
