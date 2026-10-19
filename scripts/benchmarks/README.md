# Unified NeuralOps benchmark CLI

The implementation lives exclusively in `neuralops.benchmarks`. This directory
contains only the compatibility entry point `neuralops_bench.py`; it does not
contain benchmark contracts, adapters, kernels, or reference implementations.

Examples:

```bash
python -m neuralops.benchmarks --category attention --dynamic
python -m neuralops.benchmarks --category compressor --parallel-workers 4
python -m neuralops.benchmarks --category all --dynamic \
  --output results/benchmarks_v1.4.0/master_203_FULL.json
```

Use `--modules cosformer,swiglu` to restrict discovery. The script wrapper is
equivalent to `python -m neuralops.benchmarks`; dynamic and GPU runs must be
executed on the configured remote GPU host, not on macOS.
