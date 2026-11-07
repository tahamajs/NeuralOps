# NeuralOps Project-Wide Scaling Routing

This companion map extends `.agents/SCALING_ROUTING_MATRIX.md` from agent policy files to the repository’s implementation folders. It tells an agent where a scale dimension is owned, what must be kept synchronized, and which evidence is required before moving to the next layer.

## Change propagation order

```text
config/source-of-truth
        -> workload/model builder
        -> backend dispatch/kernel
        -> test/parity matrix
        -> PBS resource launcher
        -> result schema/ledger
        -> reports/tables/README
```

Never begin at generated reports. A report-only edit is allowed only to correct a documented label or provenance error.

## Repository folder-by-folder map

| Folder | Owns | Scale changes must also inspect | Gate |
|---|---|---|---|
| `neuralops/adaptation/` | Adaptation policies and rank/shape adaptation | `config/`, `models/`, adaptation tests | shape and finite-output checks |
| `neuralops/adapters/` | Architecture adapters for BERT/GPT/Qwen/GQA | `layers/`, `surgery/`, model tests | head divisibility and forward parity |
| `neuralops/analysis/` | Profiling, sweep analysis, spectral metrics | `benchmarks/`, `results_saver/`, analysis tests | no inferred metrics from missing rows |
| `neuralops/attentions/` | Compatibility attention namespace | `layers/`, registry, compatibility tests | preserve public import paths |
| `neuralops/backward_compat/` | Legacy names and migration shims | callers via `rg`, release docs | do not remove aliases during scaling |
| `neuralops/benchmarks/` | Benchmark configs, runners, registry, resource scaling | `benchmarks/`, `scripts/benchmarks/`, benchmark tests | distinct labels and complete metadata |
| `neuralops/cli/` | User-facing benchmark/config commands | config loader, README, CLI tests | CLI help and invalid-input failure |
| `neuralops/compression/` | Canonical compressor implementations | `compression_engine/`, registry, parity tests | real compression and reconstruction metrics |
| `neuralops/compression_engine/` | Compatibility compressor namespace | `compression/`, callers | preserve compatibility; no duplicate canonical count |
| `neuralops/config/` | Typed model/data/hardware/method/sweep configuration | `tests/config/`, CLI, benchmark runners | schema validation and explicit defaults |
| `neuralops/core/` | Registries, dispatch, errors, contracts | every caller of changed symbol | registry invariants and fail-closed errors |
| `neuralops/data/` | Dataset loading, manifests, synthetic fixtures | `evaluation/`, data tests, cache docs | exact sample count and split provenance |
| `neuralops/distributed/` | TP/CP/FSDP and process coordination | `scripts/*.pbs`, distributed tests | rank/device/world-size evidence |
| `neuralops/evaluation/` | Metrics, eval patching, model evaluation | model registry, result ledger | baseline and compressed metrics both real |
| `neuralops/executor/` | Layer/device parallel execution | distributed config and executor tests | no duplicate/missing layer execution |
| `neuralops/experiment_logging/` | Run IDs, metadata, event logging | result schema and scripts | unique run ID and full scale metadata |
| `neuralops/export/` | Safetensors/ONNX/export manifests | integrations, export tests | valid artifact format; raise on failure |
| `neuralops/inference/` | Inference paths, batching, cache behavior | serving, model loaders, latency tests | synchronized timing and finite output |
| `neuralops/integrations/` | vLLM/SGLang/HF/TGI bridges | serving tests and roadmap docs | real registration or explicit unsupported state |
| `neuralops/kernels/` | Canonical kernel catalog and enumeration | `ops/cuda/`, `ops/triton/`, catalog tests | canonical counts and implementation identity |
| `neuralops/layers/` | Canonical attention implementations | adapters, registry, parity matrix | 131 logical keys and shape/causal checks |
| `neuralops/models/` | Model wrappers, surgical IO, model contracts | adapters, export, model tests | exact config/revision and round-trip state |
| `neuralops/operations/` | Quantization/pruning/weight operations | compression, kernels, operation tests | dtype, group size, sparsity, and error recorded |
| `neuralops/ops/cuda/` | Native CUDA implementations and extensions | `kernels/`, CUDA parity tests, PBS | compile/load and reference max-diff gate |
| `neuralops/ops/triton/` | Triton kernels, standalone kernels, autotune | CUDA/reference path, Triton parity tests | distinct kernel identity; no relabelled duplicate |
| `neuralops/optimization/` | Optimizers, schedules, search policies | config, trainer, optimization tests | seed/step/learning-rate metadata |
| `neuralops/pipeline_transform/` | Pipeline plans, snapshots, transformations | surgery, export, pipeline tests | topology and state-dict integrity |
| `neuralops/recipe/` | Reproducible experiment recipes | config, runners, docs | recipe pins every scale axis |
| `neuralops/reporting/` | Reports, summaries, charts, tables | results ledger and docs | report only measured rows |
| `neuralops/results_saver/` | Result persistence and schema | benchmark runners, ingestion tests | atomic writes and no hardcoded metrics |
| `neuralops/runners/` | Unified experiment orchestration | recipes, PBS scripts, runner tests | subprocess exit codes and job metadata |
| `neuralops/serving/` | Request batching and production serving | integrations, serving tests | concurrency, latency, KV cache evidence |
| `neuralops/surgery/` | Attention replacement/healing | adapters, model tests, GPU surgery suite | pre/post structure and finite output |
| `neuralops/tracking/` | Run tracking and telemetry | experiment logging, results | no collision across parallel scale points |
| `neuralops/trainer/` | Training loops, memory, checkpointing | config, optimization, trainer tests | loss/grad finite and checkpoint reproducibility |
| `neuralops/training/` | Training utilities and distributed training | trainer, distributed tests | exact batch accumulation and world size |
| `neuralops/utils/` | Shared compatibility, validation, IO helpers | all callers of changed helper | compatibility behavior covered |
| `neuralops/validation/` | Numerical/result validation gates | kernels, benchmarks, result ingestion | `FAIL_NUMERICS` on tolerance breach |
| `neuralops/validator/` | Structural/config validation | config and conformance tests | invalid scale rejected before workload |
| `benchmarks/` | Top-level reproducible benchmark programs/configs | `neuralops/benchmarks/`, scripts, results | source config and result manifest aligned |
| `benchmarks/configs/` | Named quick/exhaustive/long-context configs | resource estimator and PBS | config expands explicitly; no hidden defaults |
| `scripts/benchmarks/` | Operational benchmark entry points | source runners and PBS wrappers | no CPU fallback; logs and exit codes |
| `scripts/lib/` | Shared environment and launcher helpers | every script caller | shell syntax and environment provenance |
| `scripts/utils/` | Auditors, sync, reporting helpers | docs and manifests | audit output is raw and reproducible |
| `tests/benchmarks/` | Benchmark contract/regression tests | changed benchmark source | regression test fails before fix |
| `tests/gpu_exhaustive/` | Full GPU matrix | PBS script and model cache | real allocation and raw stdout/stderr |
| `tests/cuda_kernels/` | Native CUDA correctness | `ops/cuda/`, compiler env | compile/load/parity evidence |
| `tests/cuda_parity/` | CUDA/reference comparison | backend dispatch and tolerance config | max-diff and gradient gates |
| `tests/triton_parity/` | Triton/reference comparison | `ops/triton/`, autotune config | kernel identity and tolerance evidence |
| `tests/distributed/` | Multi-process scaling | `distributed/`, PBS launcher | all ranks, devices, and aggregation accounted for |
| `tests/config/` | Configuration scaling contracts | `neuralops/config/` | invalid ranges rejected |
| `tests/integrations/` and `tests/serving/` | Runtime/serving scaling | integrations and serving code | real engine or explicit skip, never fake pass |
| `tests/surgery/` | Model surgery scale | adapters, surgery, model cache | exact model and post-surgery parity |
| `tests/validation/` and `tests/validator/` | Validation policy | validation/validator source | missing data raises or is visibly skipped |
| `docs/` | Human-facing architecture and evidence | source, manifests, results | measured/planned/historical labels |
| `notebooks/` | Interactive demonstrations and paper experiments | benchmark source and docs | executed notebook metadata and pinned scale |
| `results/` | Generated raw results, logs, tables, figures | result schema/ledger | never hand-edit to conceal failures |
| `visualizer/` | Dashboard/UI for result exploration | reporting and result schemas | new scale axes visible and unambiguous |
| `examples/` | User-facing workflows | APIs, config, docs | example must use real backend and explicit device |
| `academic/` | Paper/thesis/proposal artifacts | evidence audit and result ledger | no unsupported scale claims |
| `assets/` | Static images and supporting assets | visualizer/docs | regenerate only from verified data |

## Scale-specific file sets

### Batch or sequence scaling

Read and search these together:

```text
neuralops/config/data_config.py
neuralops/config/hardware_config.py
neuralops/config/sweep_config.py
neuralops/benchmarks/config.py
neuralops/benchmarks/resource_scaling.py
neuralops/benchmarks/varlen.py
neuralops/benchmarks/benchmark_extrapolation_32k.py
neuralops/serving/
neuralops/inference/
tests/config/
tests/gpu_exhaustive/test_long_context_memory.py
benchmarks/configs/long_context_cuda.json
```

Required evidence: actual batch/length in every result row, peak VRAM, synchronized timing, valid `cu_seqlens`, and parity at each point.

### Precision or backend scaling

```text
neuralops/benchmarks/quantization.py
neuralops/benchmarks/measure_multiprecision.py
neuralops/operations/weights/quantize.py
neuralops/ops/cuda/
neuralops/ops/triton/
neuralops/validation/
tests/cuda_parity/
tests/triton_parity/
benchmarks/benchmark_precision_parity.py
```

Required evidence: FP32 reference, dtype, tolerance, max diff, finite values, and explicit backend status.

### Attention/compressor breadth scaling

```text
neuralops/core/registry.py
neuralops/layers/
neuralops/compression/
neuralops/kernels/
neuralops/benchmarks/registry.py
tests/gpu_exhaustive/test_all_131_attentions.py
tests/gpu_exhaustive/test_all_72_compressors.py
scripts/benchmarks/benchmark_all_cuda.py
scripts/benchmarks/benchmark_all_triton_comprehensive.py
```

Required evidence: canonical key counts, unique implementation identity, no relabelled duplicate rows, and explicit skipped/failed entries.

### Multi-GPU/distributed scaling

```text
neuralops/distributed/
neuralops/executor/
neuralops/training/
scripts/*.pbs
scripts/submit_gpu.sh
tests/distributed/
tests/test_exhaustive_distributed_tp_cp_streaming.py
```

Required evidence: `qstat -f`, allocated host/GPU map, world size, rank logs, exit status, and single-GPU parity.

### Results and documentation scaling

```text
neuralops/results_saver/
neuralops/reporting/
neuralops/experiment_logging/
results/ledger/
results/json/
PROJECT_VERIFICATION_MANIFEST.md
PROJECT_BENCHMARK_INTEGRITY_AUDIT.md
README.md
docs/
```

Required evidence: immutable run ID, source revision, exact scale metadata, raw log path, and no overwritten prior result.
