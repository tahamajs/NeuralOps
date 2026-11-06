# NeuralOps Agent Operating System

**Repository:** NeuralOps
**Applies to:** coding agents, CI bots, subagents, and human contributors
**Current package alignment:** `v2.0.6`
**Authority:** read `CONTRIBUTING.md` before changing code. This file defines
agent operations, repository boundaries, verification, and failure policy.

The rules below are operational requirements. They must describe the repository
that actually exists. If this file, a script, and the filesystem disagree, stop
and report the discrepancy; do not modify code or documentation merely to make
an expected number appear true.

## 0. Non-negotiable principles

### 0.1 Verification honesty

Never claim a command, test, kernel, compiler, benchmark, or job succeeded
unless it was physically executed and its result was observed.

Every verification report must include the exact command, working directory,
interpreter or executable path, host/device and job ID for remote work, exit
code, final summary or exact failure, and artifact/log path. An intermediate
progress line is not completion evidence. A partial pytest log, killed process,
or queued job is incomplete.

### 0.2 Zero fake artifacts and metrics

Never create placeholder files with binary or model extensions. Never write
hard-coded throughput, speedup, accuracy, perplexity, memory, or parity values
when real data is missing. If conversion, export, compilation, or evaluation
fails, raise or record an explicit failure.

The following are prohibited in data-producing paths:

```python
except Exception:
    pass
```

Also prohibited are generic SVD, zero tensors, identity matrices, default
metrics, or another implementation substituted under a failed component's
name.

### 0.3 Fail closed

Explicit requests must fail instead of changing meaning:

- `--device cuda` with unavailable CUDA -> error;
- requested native CUDA that fails -> failure or explicitly labelled fallback;
- failed named compressor -> failed compressor result, never generic SVD;
- missing or non-finite gradient -> backward failure;
- finite but numerically wrong kernel -> `FAIL_NUMERICS`;
- unavailable serving engine -> unavailable/failed status;
- incomplete or aborted exhaustive run -> incomplete/failure.

### 0.4 Preserve user state

Never run destructive commands such as `git reset --hard`, `git checkout -- .`,
or `git restore .` without explicit authorization for the exact targets. Never
discard, hide, or overwrite unrelated uncommitted work.

## 1. Read-before-edit and scope protocol

Before editing any file:

1. Read the complete file.
2. Read tests that import or exercise it.
3. Locate callers with `rg` or `git grep`.
4. Run `git status --short` and record baseline dirty files.
5. State the intended change and blast radius in one sentence.
6. Prefer the smallest diff that fixes the requested behavior.

Classify dirty files as pre-existing user work, task-owned changes, generated
artifacts, or suspicious/unclassified. Do not stage or edit pre-existing files
without an explicit reason. New tests and documentation may be added when
required. A new public module/folder under `neuralops/` requires a reason,
caller audit, API decision, and test.

## 2. Git, commits, and attribution

All commits must use:

```bash
git commit -m "<conventional message>" --author="taha <tahamaj4@gmail.com>"
```

Use conventional prefixes such as `fix(...)`, `feat(...)`, `refactor(...)`,
`docs(...)`, `test(...)`, or `chore(...)`. Do not add assistant branding or
`Co-Authored-By` trailers unless requested.

On `main`, never push without explicit authorization, never force-push, inspect
the exact staged diff, and run applicable quality gates. One logical change per
commit. Preserve history with `git mv` for actual moves.

## 3. File headers and code style

Every newly authored Python, CUDA, C++, Triton, or Bash file must begin with
the repository MIT header. Python-like files use:

```python
#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
```

For Bash, retain the shebang and use shell comments; do not put Python syntax
in a Bash file. Do not add personal author docstrings or generated-by notices.
Preserve algorithmic “why” comments and hardware workarounds unless proven
incorrect. Do not bulk-format unrelated code.

## 4. Execution environments

### 4.1 Local macOS

Allowed for static work: `rg`, `find`, `sed`, `awk`, `git grep`, `ruff`,
`bash -n`, AST inspection, documentation, and packaging inspection. CPU/MPS
dynamic tests require the session's authorization. Local macOS cannot verify
CUDA C++, Triton JIT, CUDA gradcheck, or GPU memory behavior; never label local
CPU/MPS output as GPU verification.

### 4.2 Sharif HPC

Primary remote host is `hpc` with Torque PBS and GPU hosts such as
`GPU-3090-1`.

`ssh hpc` reaches a login node. A successful SSH connection or hostname does
not prove GPU access. `nvidia-smi` must run inside a valid allocation or an
explicitly authorized direct GPU-host session.

Standard flow:

```text
local source -> rsync/scp -> PBS script in scripts/hpc/
              -> persistent session if requested
              -> unique log/results -> qstat before/after
              -> final exit code and summary
```

Do not edit source directly on the cluster as the source of truth. Do not use
another user's allocation or interfere with an active job. Every GPU report
must include host, PBS ID or direct-host authorization, Python executable,
Torch/CUDA versions, GPU name/index/memory, compiler/Triton versions, command,
log, exit code, and final summary.

### 4.3 Environment isolation

Never combine site-packages from different Conda environments through
`PYTHONPATH`. Print the selected Python path before GPU execution. For native
CUDA, record `CC`, `CXX`, `CUDAHOSTCXX`, `PATH`, `LD_LIBRARY_PATH`, `nvcc`, and
`ld`. A command that reaches the linker and fails is a native compilation
failure.

## 5. Remote jobs and long-running processes

Before a long run:

```bash
git status --short
qstat -u "$USER"
pgrep -af 'pytest|neuralops|triton'
```

Each run needs a unique output directory and log. Never share result files or
directories between processes. On a single GPU, correctness suites are
serialized unless a measured memory policy permits overlap.

For a hang: capture the last module and GPU state, terminate only the owned
process, run the case in isolation with a timeout, preserve the log/status, and
run remaining tests with any exclusion visible in the final report.

## 6. Repository invariants

Counts are logical contracts, not numbers to force. Each invariant must have an
executable authority specifying object, inclusion rules, command, expected
value, and evidence.

The documented intended values are:

```text
131 attention registry keys
72 compressor registry keys
58 canonical Triton/kernel inventory units
141 CUDA package inventory units
94 canonical notebooks
```

Static checks have shown discrepancies between these values and raw filesystem
counts, including Triton files and notebooks. Therefore these values are not
verified merely by this document. On mismatch:

```text
STOP -> show command/output -> identify logical/file distinction
      -> update manifest/verifier only with evidence
      -> do not alter registries to force a count
```

Before registry/kernel/package/notebook commits, run the current invariant
verifier and record exact output. If no verifier exists, that is a repository
gap to fix before claiming verification.

## 7. Architecture boundaries

```text
layers/              torch.nn.Module implementations
models/              model composition and checkpoint lifecycle
operations/          model mutations and validation
ops/                 CUDA, Triton, PyTorch implementations
kernels/             public kernel facade only
inference/           runtime requests, decode, and cache
integrations/        vendor adapters
recipe/              declarative schemas and serialization
pipeline_transform/  plans, snapshots, rollback, transactions
reporting/           evidence and human reports
results_saver/       persistence backends and schema validation
runners/             orchestration, scheduling, monitoring
optimization/        optimizers, objectives, schedules, optimization kernels
```

Forbidden boundaries:

- `ops` must not import CLI, runners, reporting, or result persistence;
- `layers` must not write results or import runners;
- `results_saver` must not mutate models;
- `recipe` must not execute arbitrary Python or download models;
- `operations` must mutate through model protocols and transactions;
- `integrations` must not reimplement registries;
- `runners` must not contain layer/kernel implementations.

## 8. Three-level kernel contract

Every operation follows:

```text
Level 1: native CUDA/HIP extension
Level 2: Triton JIT kernel
Level 3: vectorized PyTorch reference
```

The dispatcher records actual backend and attempts. Every fallback logs
operation, failed level, exact error, and next level. `backend_used` must never
claim CUDA when Triton or PyTorch executed.

Native CUDA is verified only when a real extension artifact is built, imported,
and executed on the target GPU. Triton operators need distinct names,
capability checks, shape/dtype constraints, PyTorch reference, and parity tests.
Do not create leaderboard rows by relabelling one kernel. A process abort is a
failure requiring isolated reproduction.

## 9. Benchmark integrity contract

Every result records run ID/key, git state, requested/actual backend, device/GPU,
dtype, shape, seed, warmup, iterations, tolerance, max diff, finite state,
metrics/resources/status/error, and Python/Torch/CUDA/Triton metadata.

Explicit CUDA unavailable raises. A failed named compressor cannot become a
generic SVD row. Missing/non-finite gradients fail. `max_diff > tolerance`
becomes `FAIL_NUMERICS`. An unavailable engine is explicit unavailable. Missing
Parquet may warn while writing valid alternate formats, or fail when required.
No shared `results/latest` writes are allowed during parallel runs.

The benchmark CLI is currently `argparse`-based in
`neuralops/benchmarks/cli.py`; Rich is optional formatting. New CLI options must
be wired into parser, config merge, handler, help, tests, and docs. A declared
but ignored flag is a bug.

## 10. CLI contract

The canonical entry points are `neuralops/benchmarks/cli.py` and
`neuralops/benchmarks/__main__.py`. Do not describe the CLI as Click/Typer
unless the implementation is actually migrated.

Rules:

- config precedence: defaults < config < environment < explicit CLI flags;
- explicit CUDA fails closed;
- `--json` stdout contains JSON only; diagnostics use stderr;
- errors have stable exit codes and actionable hints;
- plan/dry-run does not allocate large tensors or write results;
- output directories are unique or explicitly overwritten;
- help contains no unregistered command;
- completion names match the parser;
- verbose output shows effective config and backend attempts.

Recommended exit codes: `0` success, `1` benchmark failure, `2` usage/config
error, `3` backend unavailable, `4` numerical failure, `5` partial run, `6`
output/schema failure, `70` unexpected internal error.

## 11. Model surgery and transformation

Every transformation follows:

```text
load -> validate -> plan -> snapshot -> apply -> forward gate
      -> backward gate -> causal/parity gate -> save manifest -> atomic commit
```

On failure restore the snapshot and never leave a production-looking partial
checkpoint. Record target paths, operation parameters, recipe hash, validation
results, and artifact paths. KD schedules, seeds, data identity, loss weights,
and backward validation must be recorded.

## 12. Results, artifacts, and local state

Use one run layout:

```text
results/<suite>/<run_id>/
  config.json environment.json manifest.json events.jsonl results.jsonl
  summary.json logs/ artifacts/
```

SQLite, DB, Parquet, JSONL, CSV, plots, logs, checkpoints, `.pyc`, `.DS_Store`,
and compiler caches are generated state unless explicitly requested as a
checked-in fixture. They must be ignored or exported intentionally. Tests use
`tmp_path` or an explicit temporary directory. Persistence validates schema,
rejects duplicate IDs, writes atomically, and never invents metrics.

## 13. Testing tiers

### Tier 0: static

```bash
git diff --check
ruff check <changed-python-files>
bash -n <changed-shell-files>
```

### Tier 1: CPU contracts

Only when session policy permits dynamic local execution:

```bash
pytest tests/benchmarks tests/architecture -v --tb=short
```

### Tier 2: package/CLI

```bash
python -m neuralops --help
python -m neuralops.benchmarks --help
python scripts/utils/sync_version.py --check
```

### Tier 3: GPU correctness

On a real GPU, report environment and final summary:

```bash
python -m pytest tests/gpu_exhaustive -v --tb=short
```

### Tier 4: native CUDA

Require a built/imported/executed extension. Linker failure is failure.

### Tier 5: exhaustive/performance

Use unique paths, timeouts for unsafe modules, isolated processes when needed,
and an error appendix. Timeout/fatal abort remains unresolved even if earlier
cases passed.

## 14. Regression tests

Every bug fix needs a minimal test that fails before and passes after. Required
integrity cases include explicit CUDA unavailable, failed compressor, missing or
NaN gradient, duplicate Triton rows, numerical tolerance failure, ignored
config flag, duplicate run directory, missing metric, optional Parquet failure,
and timeout/incomplete status.

## 15. Security and integrations

Never print HF tokens, API keys, credential-bearing URLs, or secret environment
values. Redact snapshots and tracebacks. `trust_remote_code` must be explicit
and recorded. Validate recipe/export paths against traversal. Monitoring binds
to localhost by default. Vendor imports are lazy and handled by integration
adapters.

## 16. Release and versioning

`neuralops/__init__.py:__version__` is authoritative. Synchronize with:

```bash
python scripts/utils/sync_version.py --check
```

Release evidence:

```bash
python -m build
python -m pip install --force-reinstall dist/*.whl
python -m neuralops --version
python -m neuralops system verify --json
twine check dist/*
```

Do not release with unclassified invariant drift, fake artifacts, failed
required tests, or unreported deferred GPU verification.

## 17. Five-gate orchestrator

The repository provides `scripts/orchestrator_5gate.sh`. Read it before
describing its gates. The existence of the file is not evidence that it passed.
When required, run it and report complete output and exit code. If the tree was
dirty before the task, distinguish baseline files from task-owned files rather
than deleting or hiding them.

## 18. Required completion report

```markdown
## Executive Summary
## Baseline State
- branch:
- commit:
- dirty files before work:
## Files Modified/Created
## Verification Evidence
### Exact command
```text
<command>
```
### Exact output
```text
<output>
```
## Test Tiers
- Tier 0:
- Tier 1:
- Tier 2:
- Tier 3 GPU:
- Tier 4 native CUDA:
- Tier 5 exhaustive:
## Invariant Status
- attention registry:
- compressor registry:
- Triton inventory:
- CUDA inventory:
- notebook inventory:
## Failures and Deferred Work
## Artifacts and Logs
## Commit
```

## 19. Self-monitoring

Before every phase and remote launch:

```bash
git status --short
git diff --check
```

For remote work:

```bash
qstat -u "$USER"
pgrep -af 'pytest|neuralops|triton'
```

If an unexpected process, output collision, invariant drift, or unrelated file
change appears, stop the affected phase and report it.

## 20. Version history

- **v2.0.0** — Reconciled policy with the actual v2.0.6 repository: argparse
  CLI, verifier-based invariants, benchmark integrity, GPU evidence, artifact
  hygiene, architecture boundaries, security, and test tiers.

*Last updated: 2026-11-06.*
