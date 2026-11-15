# NeuralOps Migration Guides

Welcome to the NeuralOps migration portal. This directory provides step-by-step upgrade manuals, namespace conversion tables, copy-pasteable code recipes, and automated migration scripts to ensure smooth transitions between major and minor releases.

---

## Active Migration Guides

| Guide | Target Users | Description | Estimated Time |
| :--- | :--- | :--- | :--- |
| **[v2.3 → v2.4.0](v2.3_to_v2.4.md)** | All v2.3 users | Upgrade to the v2.4.0 release: modular `cli_v2`, Pydantic v2 `config_v2`, multi-format `results_v2` ledger, PyTorch 2.x `torch.amp`, and decoupled model surgery. | ~30 min |
| **[v1.x → v2.x](v1_to_v2.md)** | Legacy v1.x deployments | Transition from legacy monolithic flat namespaces (`attention`, `compress`, `train`) to modern modular packages (`layers`, `compression`, `trainer`). | ~2 hours |

---

## Deprecation Timeline & Lifecycle Matrix

NeuralOps strictly adheres to Semantic Versioning (SemVer 2.0.0). Deprecated APIs remain fully operational via warning-emitting backward-compatibility shims for at least one full minor cycle before scheduled removal.

| Feature / Symbol | Deprecated In | Removal Target | Canonical Replacement | Guidance Document |
| :--- | :--- | :--- | :--- | :--- |
| `continuous_batching_generate` | v2.4.0 | v2.6.0 | `ServingAdapter.generate_sequential` | [v2.3 → v2.4](v2.3_to_v2.4.md#recipe-4-upgrading-serving--batch-inference) |
| `yaml.safe_load` for benchmark configs | v2.4.0 | v2.6.0 | `config_v2.loader.load_benchmark_config` | [v2.3 → v2.4](v2.3_to_v2.4.md#recipe-2-upgrading-benchmark-configurations) |
| `neuralops.benchmarks.cli:main` | v2.4.0 | v3.0.0 | `neuralops.benchmarks.cli_v2.main:main` | [BREAKING_CHANGES.md](../../BREAKING_CHANGES.md) |
| Quarantined standalone scripts (`_legacy/`) | v2.4.0 | v3.0.0 | `neuralops benchmark run --config <path>` | [v2.3 → v2.4](v2.3_to_v2.4.md#recipe-1-upgrading-cli-benchmark-invocations) |
| `from torch.cuda.amp import ...` | v2.4.0 | v3.0.0 | `from torch.amp import autocast, GradScaler` | [v2.3 → v2.4](v2.3_to_v2.4.md#recipe-5-pytorch-2x-amp-modernization--safe-deserialization) |
| `neuralops.attentions` | v2.4.0 | v3.0.0 | `neuralops.layers` | [v1 → v2](v1_to_v2.md#complete-namespace-mapping) |
| `neuralops.compression_engine` | v2.4.0 | v3.0.0 | `neuralops.compression` | [v1 → v2](v1_to_v2.md#complete-namespace-mapping) |
| `neuralops.core.compression_engine` | v2.4.0 | v3.0.0 | `neuralops.compression` | [v1 → v2](v1_to_v2.md#complete-namespace-mapping) |
| `neuralops.training` | v2.4.0 | v3.0.0 | `neuralops.trainer` | [v1 → v2](v1_to_v2.md#complete-namespace-mapping) |
| `neuralops.core.rope` | v2.4.0 | v3.0.0 | `neuralops.core.utils.rope` | [v1 → v2](v1_to_v2.md#complete-namespace-mapping) |
| `neuralops.core.gqa_utils` | v2.4.0 | v3.0.0 | `neuralops.utils.architecture` | [v1 → v2](v1_to_v2.md#complete-namespace-mapping) |
| `neuralops.surgery.universal_calibrator` | v2.4.0 | v3.0.0 | `neuralops.trainer.regimes` | [v1 → v2](v1_to_v2.md#complete-namespace-mapping) |

---

## Automated Migration Tooling

An automated migration helper is provided in the repository to upgrade user scripts in-place:
```bash
python scripts/migrate_v23_to_v24.py path/to/your_script.py
```
This utility automatically detects deprecated function calls, replaces obsolete imports, and rewrites YAML configuration loaders.

---

## Need Assistance?

- **Issues & Backlog:** Report unexpected migration behavior in the [Issue Tracker](../../issues/).
- **Breaking Changes Catalog:** Consult [BREAKING_CHANGES.md](../../BREAKING_CHANGES.md) for complete technical commit history.
- **Troubleshooting Guide:** Common runtime anomalies and remediation recipes are indexed in [docs/TROUBLESHOOTING.md](../TROUBLESHOOTING.md).
