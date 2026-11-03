---
name: neuralops-scaling
description: Scale NeuralOps workloads, models, benchmarks, GPU jobs, and documentation through an explicit cross-file change matrix with reproducible fail-closed verification.
---

# NeuralOps Scaling Contract

Use this skill whenever the requested change increases or decreases model size, batch size, sequence length, precision, number of GPUs, benchmark breadth, queue resources, or serving throughput. Scaling is a coordinated repository change, not a single-constant edit.

## 1. Non-negotiable rules

1. Read `CONTRIBUTING.md`, `AGENTS.md`, this file, and every target file before editing.
2. Never scale by silently reducing the workload, replacing a missing model/kernel with a fake result, or falling back from CUDA to CPU.
3. Keep reference, CUDA, Triton, and PyTorch outputs under the numerical parity gate. A missing reference, NaN, Inf, or `max_diff > tolerance` is a failure.
4. GPU work runs through PBS on an allocated node. `hpc` is the login gateway; direct SSH to `GPU-3090-1` is only for read-only inspection after `qstat` confirms an active allocation.
5. Long downloads, monitoring, and control loops may use `screen`/`tmux`; benchmarks and inference must remain PBS jobs.
6. Every changed scale dimension must be recorded in the benchmark configuration, result manifest, and relevant README/doc. Do not report results from an unrecorded configuration.

## 2. Scaling dimensions and file ownership

| Scale dimension | Primary files | Required dependent files | Verification |
|---|---|---|---|
| Model family/checkpoint | `neuralops/benchmarks/registry.py`, benchmark CLI/config | model loaders, cache manifest, README, result manifest | local-files-only resolution, exact model ID, finite outputs |
| Hidden size/layers/heads | benchmark configuration and model fixture | attention adapter, shape contracts, memory estimates | shape assertions, GQA head divisibility, forward/backward |
| Batch size | `neuralops/benchmarks/config.py`, suite CLI | memory estimator, result schema, sweep manifest | measured peak VRAM and OOM classified as failure |
| Sequence length/context | benchmark config and workload generator | causal masks, varlen packing, KV-cache sizing | `cu_seqlens` boundaries, parity at every length |
| Precision/dtype | benchmark precision config and kernel dispatch | CUDA/Triton guards, tolerance table, result schema | FP32 reference, dtype-specific tolerance, finite gradients |
| Attention/compressor breadth | registries and canonical catalog | benchmark runner, static catalog, coverage tests | canonical key counts; no relabelled duplicate rows |
| GPU count/distributed scale | PBS script, distributed launcher | rank/world-size setup, rendezvous, result aggregation | `qstat -f`, host/GPU map, all ranks accounted for |
| GPU memory/CPU/time | PBS resource directives | preflight checks and manifest | requested vs allocated resources and walltime |
| Serving throughput/concurrency | serving benchmark modules | engine integration contract and telemetry schema | real engine registration; unsupported integration skips or raises |
| Download/model cache scale | download script and cache manifest | screen command, log path, completed-model ledger | exact bytes/files, retry error, no official endpoint bypass |

## 3. Required change procedure

### Phase A — Baseline

Record:

```text
git status --short
git diff --stat
model/checkpoint and revision
device, CUDA/PyTorch/Triton versions
dtype, batch size, sequence lengths
seed(s), warmup, measured iterations
PBS queue, job ID, allocated host/GPU
```

Inspect callers before changing a shared symbol:

```bash
rg -n "<symbol>|<config_key>|<model_id>" neuralops scripts benchmarks tests docs
```

### Phase B — Change the source of truth

Change the narrowest authoritative configuration first. Do not edit generated results or README numbers before the source configuration and schema are correct. If a new scale axis is needed, add it to the existing configuration/result schema rather than duplicating a second global constant.

### Phase C — Propagate the scale

Update all consumers listed by the matrix above. In particular:

- workloads must carry the scale values into each result row;
- memory estimates must use the actual dtype, batch, length, heads, and KV heads;
- benchmark labels must identify every distinct configuration;
- result aggregation must reject missing or duplicate configurations;
- docs must distinguish measured, planned, skipped, and failed states;
- PBS scripts must use `cd "$PBS_O_WORKDIR" || exit 1` and the intended absolute interpreter.

### Phase D — Verify in increasing cost order

1. Static: `ruff check <changed_files>` and `git diff --check`.
2. CPU/static contract tests where allowed by `AGENTS.md`.
3. One GPU smoke case on the smallest valid scale.
4. Full GPU matrix on PBS.
5. Multi-GPU/distributed scale only after single-GPU parity passes.
6. Record raw PBS output, job state before/after, and result manifest.

## 4. GPU/PBS scaling contract

Use the configured multiplexed SSH alias and avoid repeated independent sessions:

```bash
ssh -O check hpc
ssh hpc 'qstat -u "$USER"'
cd /path/to/NeuralOps
qsub scripts/<job>.pbs
qstat -n <JOB_ID>
qstat -f <JOB_ID>
```

Submit from the project directory so `PBS_O_WORKDIR` is correct. A valid GPU run must show an allocated host such as `GPU-3090-1/0`; `Q` is queued, `R` running, and `C` completed. A completed job is not evidence of success without its stdout/stderr and exit status.

For disconnected monitoring only:

```bash
ssh -t hpc 'screen -r <session>'
```

Do not put benchmark execution in a login-node screen. Use screen for lightweight monitoring or resumable downloads; use PBS for GPU compute.

## 5. Numerical and performance scaling gates

For each scale point, store:

```json
{
  "model": "exact-id-or-local-revision",
  "device": "cuda:0",
  "dtype": "float16",
  "batch_size": 1,
  "sequence_length": 1024,
  "max_diff": 0.0,
  "tolerance": 0.001,
  "status": "PASS"
}
```

The status must be `FAIL_NUMERICS` when `max_diff > tolerance`, even if all values are finite. Missing gradients, non-finite gradients, missing kernels, failed compressors, and unavailable requested CUDA are hard failures. Never convert those states to `PASS`, `SKIP`, or a generic fallback result without an explicit environment-dependency reason.

Performance claims require measured warmup/iteration counts, synchronization around timing, peak memory, and the exact hardware/job ID. Never extrapolate a larger scale from a smaller measurement and label it measured.

## 6. Multi-GPU and distributed changes

Before increasing `world_size`:

1. Validate the same workload on one GPU.
2. Verify rank-local device assignment and `CUDA_VISIBLE_DEVICES`.
3. Verify rendezvous/address configuration without embedding credentials or tokens.
4. Ensure every rank writes a unique temporary/log path and only rank zero writes the aggregate result.
5. Compare distributed output against the single-GPU reference within the documented tolerance.
6. Cancel orphaned jobs with `qdel` when the user authorizes cleanup.

Do not claim multi-GPU scaling from multiple sequential single-GPU jobs.

## 7. Model and download scaling

Model downloads are separate from benchmark execution. Use the authorized mirror and a persistent screen session, keep a completed-model ledger, and report exact failures. A gated or unavailable model is not replaced with a smaller model under the same label. If the mirror returns an upstream block, record it and stop/retry according to the download policy; do not bypass authentication or use an unapproved endpoint.

## 8. Documentation and result synchronization

When scale changes, update the smallest authoritative docs plus the manifest:

- `AGENTS.md` for repository invariants and operational rules;
- `PROJECT_STRUCTURE.md` / dependency maps when file ownership changes;
- `PROJECT_BENCHMARK_INTEGRITY_AUDIT.md` for integrity status;
- `PROJECT_VERIFICATION_MANIFEST.md` for command, job, raw result, and status;
- the relevant README/roadmap only after measured evidence exists.

Historical numbers must be labeled historical. Planned numbers must be labeled planned. Skipped and failed configurations must remain visible; deleting rows to make a chart look complete is prohibited.

## 9. Completion checklist

```text
[ ] Source configuration changed, not only generated output
[ ] All callers and dependent schemas updated
[ ] Distinct scale points have distinct labels and result rows
[ ] CUDA request fails closed when unavailable
[ ] No fake compressor/kernel/model fallback exists
[ ] Numerical parity gate checks max_diff against tolerance
[ ] ruff check passes for changed Python files
[ ] GPU job ran through PBS on an allocated host
[ ] qstat before/after and full stdout/stderr captured
[ ] Results include hardware, dtype, batch, length, seed, and job ID
[ ] README/manifest/docs match measured evidence
[ ] Unrelated user changes preserved
```

