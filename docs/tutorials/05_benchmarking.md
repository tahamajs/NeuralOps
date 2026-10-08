# Benchmarking Suite Tutorial

Profile latency, peak memory, throughput, and numerical accuracy:

```python
from neuralops.benchmarks.suite_runner import run_and_save_benchmark_suite

csv_file, md_file = run_and_save_benchmark_suite("benchmarks")
```
