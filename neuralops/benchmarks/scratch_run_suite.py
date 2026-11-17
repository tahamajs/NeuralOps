from neuralops.benchmarks.suite_runner import run_and_save_benchmark_suite
import time

t0 = time.time()
run_and_save_benchmark_suite()
print(f"Suite finished in {time.time() - t0:.2f}s")
