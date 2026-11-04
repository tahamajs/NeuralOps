# Verification Manifest

This manifest records the strongest evidence currently available for the
repository-wide audit. It deliberately separates static evidence from remote
runtime evidence.

| Gate | Scope | Evidence | Status |
|---|---|---|---|
| Ruff | Entire local repository | `./.venv/bin/ruff check .` → `All checks passed!` | PASS |
| Whitespace/error check | Entire local diff | `git diff --check` → no output | PASS |
| Inventory | Tracked repository | `git ls-files` → 13,318 files; top-level directory inventory recorded | PASS |
| Caller map | Registries, compatibility namespaces, catalogs, benchmark entrypoints, dynamic imports | `PROJECT_CALLER_MAP.md` with confirmed search evidence | PASS (static) |
| Benchmark integrity regression coverage | GPU device fail-closed, compressor failures, gradients, parity mismatch | `tests/benchmarks/test_gpu_benchmark_integrity.py` | PRESENT; runtime not executed locally |
| Canonical count documentation | 131/72/26/62/110/114/29 | AGENTS, README, roadmap, catalog docs reconciled | PASS (static) |
| Remote queue availability | Sharif HPC | Read-only `ssh hpc`: N3090 enabled; prior user jobs completed | OBSERVED |
| Remote GPU test suite | `tests/gpu_exhaustive` | Job 893943 on `GPU-3090-1`: `894 passed, 7 failed, 14 skipped in 1091.10s` | FAIL (7 actionable failures) |

## Runtime verification requirement

The final runtime gate must be run from an allocated `N3090` PBS job using the
project's `neuralops` environment. The evidence must include the job ID,
`qstat -f` before and after, complete stdout/stderr, and the pytest summary.
Until those artifacts exist, this repository is not described as GPU-certified.

## Remote failure evidence

- `893940`: `/var/spool/torque/mom_priv/jobs/893940.hpc.ce.sharif.edu.SC: line 21: CUDA_VISIBLE_DEVICES: PBS did not allocate a CUDA device`.
- `893941`: PyTorch reported `The NVIDIA driver on your system is too old (found version 11030)` while using the `neuralops` CUDA 12.1 environment.
- `893942`: `/home/tahamajlesi.ut/.conda/envs/neuralops_cu113/bin/python: No module named pytest`.
- `893943`: selected `neuralops_cu113`, added GCC/CUDA library paths, installed pytest, and ran on RTX 3090. Final result: `894 passed, 7 failed, 14 skipped in 1091.10s`.

## Job 893943 failure classification

- 2 failures: `named_parameters(remove_duplicate=False)` compatibility with the
  installed Transformers/Accelerate stack.
- 3 failures: Qwen model configuration/weights unavailable in the offline cache;
  these are environment/model-portfolio failures, not silently treated as passes.
- 2 failures: missing `register_neuralops_vllm_ops` and
  `register_neuralops_sglang_ops` exports in the integration namespaces.

## Worktree preservation

Existing user modifications remain uncommitted and untouched. No source,
compatibility module, benchmark, or test was deleted during this audit.
