Module neuralops.benchmarks.run_seqlen_scaling
==============================================
Sequence-Length Scaling Runner: Evaluates VRAM, latency, and throughput scaling across context lengths.

Evaluates scaling across sequence lengths: [128, 512, 1024, 2048, 4096]
Modules: ["dense", "a3", "kmeans", "svd-mamba", "primal"]

Measures:
  - Peak VRAM footprint via torch.cuda.max_memory_allocated()
  - Forward pass latency (ms) and throughput (tokens/sec)
  - Graceful OOM handling: marks "OOM" in results without crashing sweep

Usage:
    python run_seqlen_scaling.py --dry_run
    python run_seqlen_scaling.py --batch_size 2

Functions
---------

`benchmark_forward_scaling(model, batch_size: int, seq_len: int, warmup: int = 3, steps: int = 10, device: str = 'cuda')`
:   Executes warmup and timed forward passes with synthetic sequence inputs.
    Returns peak VRAM (MB), latency (ms), and throughput (tok/s).

`main()`
:   

`parse_args()`
: