# NeuralOps configuration schemas

`neuralops.config` exposes typed loaders. Use the loader that matches the file's
schema; the loaders reject unknown keys rather than silently dropping settings.

| File | Intended schema | Loader |
|---|---|---|
| `default.yaml` | Flat CLI experiment settings | `load_experiment_config` |
| `sweep_attention_types.yaml` | Attention sweep settings (also accepts compatible flat CLI presets) | `load_attention_sweep_config` |
| `qwen3_5.yaml` | Qwen-specific quality sweep settings | `load_quality_sweep_config` |
| `ablation_pipeline.yaml` | Ordered stages of an ablation | `load_ablation_pipeline` |
| `blocklist.txt` | CSV module policy records | `load_blocklist` |

The canonical nested dataclass tree is `Config`, composed of `ModelConfig`,
`DataConfig`, `TrainingConfig`, `CompressionConfig`, `MethodConfig`,
`HardwareConfig`, `SweepConfig`, and `ExperimentConfig`. Load it with
`load_config("path/to/nested-config.yaml")`. A flat CLI preset is a distinct
schema and should be loaded with `load_experiment_config`; passing it to
`load_config` now raises a useful unknown-field error.

`blocklist.txt` is a documented benchmark policy, while `qwen3_5.yaml` carries
Qwen-run-specific skips and timeouts. They are intentionally not merged: their
scopes differ. Likewise, the Qwen quality sweep and ablation pipeline are
validated configuration documents, not implicitly executed by importing the
package. The ablation loader validates stage structure and order; it does not
resolve module aliases or execute stages. Scripts should opt into their
matching loader explicitly and resolve module names against the appropriate
registry before execution.

`ExperimentConfig` also retains compatibility fields used by `neuralops.main`
and the dataset loader (`model_name`, `task`, `max_len`, `lr`, `grad_accum`,
`dataset_name`, and related CLI settings). The nested `Config` remains the
structured API for new programmatic configurations.
