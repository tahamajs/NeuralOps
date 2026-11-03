# NeuralOps Benchmark Cleanup and Bug Audit

## Purpose

This document is the implementation contract for cleaning `benchmarks/`,
`neuralops/benchmarks/`, and `scripts/benchmarks/`. It describes the target
architecture, the defects that have already been observed, the additional
failure modes that must be audited, and the evidence required before the
cleanup can be considered complete.

The goal is not to delete historical experiments blindly. The goal is to make
one trustworthy benchmark system with explicit compatibility wrappers for
legacy experiments.

## Current problem: three overlapping systems

The repository currently contains three layers with overlapping names and
different contracts:

1. `neuralops/benchmarks/` contains the package implementation, discovery,
   orchestration, adapters, metrics, and result writers.
2. `benchmarks/` contains root-level entry points, newer canonical probes, JSON
   presets, and compatibility wrappers.
3. `scripts/benchmarks/` contains older GPU suites, exporters, visualizers,
   historical paper experiments, and several independent argument schemas.

This is not automatically wrong. It becomes dangerous when two files claim to
run the same benchmark but use different defaults, devices, result schemas, or
fallback behavior.

## Target directory layout

```text
benchmarks/
  README.md                         # user-facing quick start
  ARCHITECTURE.md                   # short data-flow reference
  BENCHMARK_CLEANUP_AND_BUG_AUDIT.md# this operational contract
  benchmark_catalog.json            # machine-readable ownership map
  configs/                          # JSON/YAML canonical presets only
  run_suite.sh                      # one stable shell dispatcher
  benchmark_<name>.py               # thin compatibility entry points only

neuralops/benchmarks/
  config.py                         # canonical configuration
  cli.py                            # canonical CLI
  suite_runner.py                   # suite orchestration
  discovery.py / registry.py        # module discovery
  adapters/                         # model/module adapters
  metrics/                          # correctness and performance metrics
  results/                          # typed schema and serializers
  <benchmark>.py                    # actual benchmark implementations

scripts/benchmarks/
  README.md                         # legacy policy and migration table
  legacy/                           # historical experiments after migration
  benchmark_config.py               # compatibility adapter only
  <old-script>.py                   # retained only when it cannot yet migrate
```

`results/`, `docs/`, `audit/`, `logs/`, `paper/`, and other generated or
academic directories are not benchmark source directories and must not be used
as hidden storage for code.

## Ownership rule

Every benchmark must have exactly one owner:

| Layer | Allowed responsibility |
|---|---|
| `neuralops/benchmarks/` | Implementation, typed result creation, validation |
| `benchmarks/` | Stable command-line wrapper and catalog entry |
| `scripts/benchmarks/` | Legacy-only compatibility or migration tooling |

A root wrapper must not duplicate benchmark logic. A legacy script must either
call the canonical implementation or be explicitly marked historical and
excluded from the standard suite.

## Canonical execution contract

Every new benchmark must follow this sequence:

```text
config file / CLI
        -> BenchmarkConfig
        -> explicit device validation
        -> deterministic seed setup
        -> benchmark implementation
        -> numerical and finite-value gates
        -> typed BenchmarkResult
        -> JSONL + JSON + CSV/Markdown exporters
```

The benchmark must record at least:

- `run_id`, `run_key`, timestamp, git commit, and dirty-tree state;
- module name, category, model/dataset identity;
- device, CUDA version, dtype, shape, seed, warmup, and iteration count;
- measured metrics and resource measurements;
- status and exact error text when the run fails.

## Bugs already found

### 1. Silent CPU fallback

Some paths accepted `--device cuda`, discovered that CUDA was unavailable, and
continued on CPU. This invalidates every latency, memory, and throughput claim.

Required behavior:

```python
if requested_cuda and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested ... but CUDA is unavailable")
```

The result must not be marked `success`, and no GPU-labelled artifact may be
written.

### 2. Fake compressor success

The master suite substituted a generic SVD approximation when a named
compressor failed to instantiate or did not implement `compress()`. That made
the result appear to belong to the requested compressor although it never ran.

Required behavior: record the exact exception and mark the module failed. Never
replace a failed implementation with another algorithm under the same name.

### 3. Forced backward-pass success

An `or True` condition made the backward validation pass regardless of gradient
state. A valid check must reject both missing and non-finite gradients:

```python
if parameter.grad is None:
    raise AssertionError("missing gradient")
if not torch.isfinite(parameter.grad).all():
    raise AssertionError("non-finite gradient")
```

The check must cover every trainable parameter participating in the test.

### 4. Fake Triton leaderboard

The Triton leaderboard executed one implementation repeatedly and relabelled
the rows as 26 kernels. This is not coverage or a leaderboard.

Required behavior: resolve 26 distinct canonical modules and run each by its
real name, or raise `NotImplementedError` listing the missing kernels. A row
must never be created for a kernel that did not execute.

### 5. Missing numerical parity gate

NaN checks alone do not prove correctness. Triton/CUDA output must be compared
with the PyTorch reference:

```text
max_diff = max(abs(candidate - reference))
max_diff <= tolerance -> PASS
otherwise             -> FAIL_NUMERICS
```

The tolerance must be recorded with dtype and shape. A finite but incorrect
output is still a failure.

### 6. Configuration split-brain

The package configuration and the legacy script configuration used different
field names and defaults. In particular, `dtype` versus `precision`, output
paths, warmup/iteration fields, and category selection could silently diverge.

New code must use `neuralops.benchmarks.config.BenchmarkConfig`. The legacy
model may only adapt into the canonical model at one explicit boundary.

### 7. Declared `--config` ignored by CLI

The CLI exposed a config option but previously did not consistently apply its
values to the run. This creates the most dangerous kind of reproducibility bug:
the command appears configured while using defaults.

Every effective configuration must be serialized beside the result and tested
with a preset regression test.

### 8. Result-schema fragmentation

The package result schema, runner writers, and historical exporter can produce
different JSON/CSV shapes. A flat CSV cannot be merged into the master ledger
without schema validation.

All canonical runs must use the typed result saver. Legacy exporters must write
to a legacy namespace and must not silently masquerade as canonical records.

### 9. Direct-file import shadowing

Running a file from `benchmarks/` can put that directory first on `sys.path`.
The local `logging.py` then shadows Python's standard-library `logging` module.

The standard runner must invoke package modules (`python -m ...`) or use a
safe wrapper that preserves repository import semantics. Direct-file execution
must be tested separately.

### 10. Optional Parquet dependency treated as success

When `pyarrow` and `fastparquet` are missing, Parquet export fails. The system
must not create a fake `.parquet` file or claim that Parquet was written.

Allowed policies are:

- write valid JSON/JSONL/CSV/Markdown and emit an explicit warning that Parquet
  is unavailable; or
- fail a run configured with `require_parquet=true`.

The selected policy must be visible in the result metadata.

### 11. GPU compiler/linker mismatch

The HPC GCC 8.2 compiler was found, but its matching `ld` was absent while the
system linker was GNU ld 2.20.51. GCC then failed while linking a CUDA extension
with the `-plugin` option.

This is an environment blocker, not evidence that native CUDA code compiled.
The benchmark must record native compilation as unavailable and may fall back
through the documented CUDA -> Triton -> PyTorch hierarchy with warnings.

### 12. Wrong Python environment on HPC

The `neuralops` environment and `neuralops_cu113` environment contain different
Torch/CUDA stacks. Mixing their `site-packages` through `PYTHONPATH` caused
driver and extension mismatches.

GPU jobs must print and record:

```text
python executable
torch.__version__
torch.version.cuda
torch.cuda.get_device_name(0)
```

No foreign environment site-packages directory may be appended manually.

### 13. Direct GPU and scheduler confusion

`ssh hpc` reaches a login node, not a GPU node. `GPU-3090-1` is reachable from
the HPC network and exposes an RTX 3090, but direct access is an operational
exception and must be used only with explicit authorization. Standard runs
should remain scheduler-managed; direct runs must record the hostname and GPU
identity.

### 14. Duplicate concurrent runs writing one artifact

Two exhaustive processes were found writing the same `final_exhaustive.log`.
This corrupts evidence and makes pass/fail counts unreliable.

Every run needs a unique output directory or job-specific filename. A runner
must refuse to reuse an active `run_id` unless resume was explicitly requested.

### 15. Incomplete exhaustive output mistaken for completion

An aborted pytest process may leave hundreds of passing lines but no summary.
The run is incomplete until the log contains pytest's final summary and exit
status. A fatal process abort is a failure, not a skip.

### 16. Registry/file-count confusion

Canonical counts are logical registry keys, not implementation-file counts.
The documented 131 attentions and 72 compressors must be checked against the
registries/static catalog, not against raw `.py` file counts.

### 17. Causal masking failures hidden by process abort

Triton compiler aborts can terminate Python before pytest reports a normal
failure. Causal-mask tests must have a safe reference path and isolated process
execution for unstable kernels. A process abort must remain a recorded failure.

### 18. Precision aliases and unsupported dtypes

`fp32`, `float32`, `fp16`, and `float16` must resolve consistently. Unsupported
precision names must raise a clear error before allocating tensors.

### 19. Reproducibility without deterministic metadata

Two equal outputs are not enough. The run must record the seed, device, dtype,
determinism settings, software versions, and whether exact equality or a
tolerance-based comparison was used.

### 20. Hard-coded metrics and empty-data fallbacks

Missing baseline accuracy, throughput, or retention values must raise or be
marked unavailable. They must never be replaced with a plausible constant,
zero, identity matrix, or empty success record.

## Cleanup procedure

### Phase A: inventory

Create a table for every benchmark file with:

- owner layer;
- category;
- public CLI;
- implementation target;
- result writer;
- tests;
- status: canonical, migrate, legacy, or delete-after-review.

No file is removed solely because its name looks duplicated.

### Phase B: choose canonical owners

For each duplicated benchmark name, select one implementation in
`neuralops/benchmarks/`. Convert root files into thin wrappers. Mark scripts that
cannot migrate immediately as legacy and document their reason.

### Phase C: normalize interfaces

Use one config contract, one device validator, one seed initializer, one timing
utility, one parity gate, and one typed result saver. Do not copy these helpers
into individual benchmark files.

### Phase D: isolate generated artifacts

Every invocation gets:

```text
results/<suite>/<run_id>/
  config.json
  environment.json
  results.jsonl
  summary.json
  logs/<job>.log
```

No benchmark may write directly to a shared `results/latest.json` during a
parallel run.

### Phase E: add regression coverage

Minimum tests:

- CUDA requested but unavailable raises;
- failed compressor cannot produce a success row;
- missing/NaN gradient fails;
- duplicate Triton module names are rejected;
- numerical difference above tolerance becomes `FAIL_NUMERICS`;
- config preset changes effective configuration;
- missing result metric raises rather than defaulting;
- active run IDs cannot collide;
- optional Parquet failure does not create a fake file.

For each bug fix, prove fail-before/pass-after in an isolated regression test.

### Phase F: verification

Run, in order:

```bash
ruff check <changed-files>
bash -n benchmarks/run_suite.sh scripts/*.pbs
git diff --check
python3 scripts/utils/audit_dead_code.py
python3 scripts/audit_readme_consistency.py
```

Then run CPU-safe contract tests and GPU suites on the correct remote
environment. A GPU run is accepted only with a final pytest summary, exit code,
hostname, GPU identity, Python path, Torch version, and output directory.

## Definition of done

The cleanup is complete only when:

- every standard benchmark has one canonical owner;
- root wrappers contain no copied benchmark logic;
- legacy scripts are explicitly classified;
- all standard runs use canonical config and typed results;
- no silent CPU fallback or fake numerical result exists;
- parallel runs have isolated artifacts;
- lint and repository audits pass;
- regression tests demonstrate fail-before/pass-after;
- exhaustive GPU output has a final summary with no unclassified failures;
- native CUDA status is either proven compiled or explicitly recorded as
  unavailable with the linker evidence;
- the final commit contains only the reviewed benchmark cleanup.

Until every item above has evidence, the project must be reported as
`incomplete`.
