# NeuralOps operational scripts

The supported entry points are intentionally small and composable:

```bash
# Local or already-allocated GPU node
bash scripts/run_gpu.sh validate-cuda
bash scripts/run_gpu.sh benchmark --quick
bash scripts/run_gpu.sh triton --quick
bash scripts/run_gpu.sh engine --engine pytorch

# Submit through PBS; --screen keeps the submission alive after disconnecting
bash scripts/submit_gpu.sh --screen --name cuda_validation \
  --command 'scripts/run_gpu.sh validate-cuda'
bash scripts/submit_gpu.sh --screen --name benchmark_suite \
  --walltime 24:00:00 --command 'scripts/run_gpu.sh benchmark --export'
```

`lib/neuralops_env.sh` is the shared environment bootstrap. Set
`NEURALOPS_PYTHON` when the interpreter is not `.venv/bin/python`.

Operational helpers:

```bash
bash scripts/doctor.sh
bash scripts/run_exhaustive_gpu.sh
bash scripts/monitor_gpu_job.sh 893842.hpc.ce.sharif.edu results/logs/job.log
bash scripts/collect_gpu_report.sh
```

`run_exhaustive_gpu.sh` refuses to run without an allocated CUDA device unless
`NEURALOPS_ALLOW_CPU_EXHAUSTIVE=1` is explicitly set. `monitor_gpu_job.sh` is
safe to leave running in a detached `screen` session.

Use `scripts/benchmarks/README.md` for the benchmark-specific entry points and
the canonical `neuralops.benchmarks` package commands. Generated outputs belong
under `results/`, which is intentionally not part of source control.
