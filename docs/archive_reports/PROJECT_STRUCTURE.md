# NeuralOps Project Structure

This document is the navigation guide for the repository. It describes where production code, kernels, benchmarks, tests, documentation, and generated artifacts belong.

## Repository map

```text
.
├── neuralops/        Core Python package and public APIs
├── tests/            Unit, integration, benchmark-integrity, and GPU tests
├── scripts/          Repository utilities, benchmark entry points, audits, and HPC helpers
├── benchmarks/       Standalone benchmark configurations and model assets
├── examples/         Small runnable examples and recipes
├── notebooks/        Reproducible exploratory and demonstration notebooks
├── docs/             Developer, API, benchmark, integration, and user documentation
├── assets/           Static project assets
├── .agents/          Repository-local agent skills and workflows
├── .github/          CI, issue, and repository automation configuration
├── pyproject.toml    Package metadata, dependencies, and tool configuration
├── CONTRIBUTING.md   Contributor workflow and coding conventions
├── AGENTS.md         Agent-specific operating and verification rules
├── PROJECT_FILE_INVENTORY.md
│                     Static file ownership and source-of-truth inventory
├── PROJECT_DEPENDENCY_MAP.md
│                     Verified execution and cleanup dependency relationships
├── PROJECT_CLEANUP_CANDIDATES.md
│                     Evidence ledger for possible consolidation or removal
├── PROJECT_DOCUMENTATION_MATRIX.md
│                     Documentation ownership and consistency matrix
├── BASELINE_STATE.md  Audit baseline and preservation rules
├── PROJECT_COMPLETE_DEPENDENCY_GRAPH.md
│                     Cross-file dependency audit record
├── PROJECT_CLEANUP_MATRIX.md
│                     Evidence-based cleanup decision matrix
└── README.md         User-facing project overview and quick start
```

## Source package

The `neuralops/` package is organized by responsibility:

- `layers/` and `attentions/`: attention implementations grouped by family.
- `compression/` and `compression_engine/`: compression algorithms and compatibility layers.
- `core/`: registries, backend dispatch, autograd utilities, fused operations, and canonical CUDA/Triton kernel catalogs.
- `kernels/` and `ops/`: backend-facing kernel implementations and PyTorch/CUDA/Triton operations.
- `benchmarks/`: reusable benchmark infrastructure, task definitions, metrics, model adapters, and benchmark utilities.
- `analysis/`, `evaluation/`, and `reporting/`: experiment analysis, evaluation workflows, and result/report generation.
- `adaptation/`, `surgery/`, `pipeline_transform/`, and `operations/`: model transformation and attention/compression workflows.
- `models/`, `inference/`, `serving/`, and `integrations/`: model-facing APIs and external runtime integrations.
- `config/`, `data/`, `executor/`, `runners/`, and `experiment_logging/`: configuration, datasets, execution, sweep orchestration, and telemetry.
- `export/`, `results_saver/`, `recipe/`, `optimization/`, and `cli/`: export, persistence, recipes, optimization, and command-line interfaces.

## Benchmarks and verification

Use the narrowest matching location:

- `scripts/benchmarks/`: standalone command-line benchmark programs and strict integrity checks.
- `neuralops/benchmarks/`: importable benchmark components used by the package and test suite.
- `tests/benchmarks/`: CPU-safe benchmark-contract and regression tests.
- `tests/gpu_exhaustive/`: remote GPU-only exhaustive validation; submit through PBS on `N3090`.
- `benchmarks/configs/` and `benchmarks/models/`: benchmark inputs and model metadata/assets.

GPU and Triton execution must not run on the login node or local macOS. Follow `AGENTS.md` and `.agents/skills/hpc-sharif-cluster/SKILL.md` for remote execution, environment selection, PBS allocation, and persistent monitoring.

## Documentation and generated data

- Root Markdown files describe project policy, setup, contribution, security, and high-level status.
- `docs/` contains extended documentation, API material, guides, integrations, benchmark notes, and historical reports.
- `results/`, `logs/`, `audit/`, and other generated-output directories are experiment artifacts; follow `.gitignore` and `AGENTS.md` before adding or committing files.
- Do not commit model weights, credentials, tokens, raw GPU logs, or generated benchmark output unless a repository policy explicitly requires the artifact.

## Change placement rules

1. Put reusable production behavior under the appropriate `neuralops/` package rather than under `scripts/`.
2. Put one-off orchestration, audits, and command-line utilities under `scripts/`.
3. Put tests under `tests/` and keep GPU-dependent tests separate from CPU-only tests.
4. Update the nearest relevant documentation when a public path, benchmark contract, environment, or execution workflow changes.
5. Preserve the registry and kernel-count invariants documented in `AGENTS.md`; distinguish catalog counts from raw filesystem counts.

The detailed audit records are maintained in `BASELINE_STATE.md`, `PROJECT_FILE_INVENTORY.md`, `PROJECT_DEPENDENCY_MAP.md`, `PROJECT_CALLER_MAP.md`, `PROJECT_COMPLETE_DEPENDENCY_GRAPH.md`, `PROJECT_CLEANUP_CANDIDATES.md`, `PROJECT_CLEANUP_MATRIX.md`, `PROJECT_DOCUMENTATION_MATRIX.md`, and `PROJECT_VERIFICATION_MANIFEST.md`. They must be updated when a package is moved, split, merged, deprecated, or removed.

## HPC execution path

The standard remote workflow is:

```text
Mac agent
  └─ SSH alias hpc (login gateway; multiplexed connection)
       └─ PBS queue N3090 (GPU allocation)
            └─ GPU-3090-1 (assigned compute node)
                 └─ $HOME/.conda/envs/neuralops/bin/python
```

The login gateway is not the GPU execution host. A compute-node shell is valid for GPU work only after PBS confirms an active allocation for the job.
