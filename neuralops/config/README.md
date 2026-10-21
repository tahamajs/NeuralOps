# NeuralOps configuration schemas

`neuralops.config` exposes typed loaders. Use the loader that matches the file's
schema; the loaders reject unknown keys rather than silently dropping settings.

| File | Intended schema | Loader |
|---|---|---|
| `default.yaml` | Flat CLI experiment settings | `load_experiment_config` |
| `sweep_attention_types.yaml` | Attention sweep settings and compatible flat CLI presets | `load_attention_sweep_config` |

The canonical nested dataclass tree is `Config`, composed of `ModelConfig`,
`DataConfig`, `TrainingConfig`, `CompressionConfig`, `MethodConfig`,
`HardwareConfig`, `SweepConfig`, and `ExperimentConfig`. Load it with
`load_config("path/to/nested-config.yaml")`. A flat CLI preset is a distinct
schema and should be loaded with `load_experiment_config`; passing it to
`load_config` now raises a useful unknown-field error.

`ExperimentConfig` also retains compatibility fields used by `neuralops.main`
and the dataset loader (`model_name`, `task`, `max_len`, `lr`, `grad_accum`,
`dataset_name`, and related CLI settings). The nested `Config` remains the
structured API for new programmatic configurations.
