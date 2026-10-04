"""
NeuralOps Benchmarks & Sweepers Suite.

Sweep drivers, scaling analysis, multi-seed evaluation harnesses, and
GPU profiling utilities for the full NeuralOps benchmark protocol.

Modules
-------
run_all_benchmarks
    Runs the complete benchmark matrix for all registered modules.
run_gpt2_sweep
    GPT-2-specific quality/speed sweep.
run_long_context_sweep
    Scaling analysis over sequence lengths 512 – 16384.
run_multiseed
    Multi-seed robustness evaluation.
run_multiseed_rigor
    Rigorous multi-seed evaluation with statistical testing.
run_one_module_isolated
    Isolated single-module benchmark for debugging.
run_quality_sweep
    Quality-vs-rank Pareto frontier sweep.
run_rank_sweeper
    Rank ablation sweep across all registered modules.
run_ruler_benchmark
    RULER long-context benchmark driver.
run_seqlen_scaling
    Sequence-length scaling experiment.
run_seqlen_scaling_bench
    Sequence-length scaling with hardware profiling.
run_whitened_svd
    Whitened SVD compression benchmark.
launch_parallel_sweep
    Parallel multi-GPU sweep launcher.
run_all_modules_scheduler
    Priority-queue-based module scheduler.
kd_quick_test
    Quick knowledge-distillation sanity test.
gpu_live_probe
    Real-time GPU telemetry probe (run on GPU server).
harvest_gpu_benchmark_data
    Post-hoc GPU benchmark data harvesting.

Subpackages
-----------
core
    Shared argument parsing (``arguments.py``) and stats utilities (``stats.py``).
module_sweeps
    Not here — per-module ``run_all_4_*.py`` scripts live in
    ``neuralops.runners.module_sweeps``.
"""
