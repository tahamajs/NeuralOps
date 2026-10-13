"""
Example: Benchmark Sweep via Python API

Shows how to:
  1. Configure a multi-module benchmark sweep via Runner.
  2. Stream execution results as modules are evaluated.
  3. Inspect aggregated benchmark summary DataFrame.

Requires:
  - neuralops installed (pip install neuralops)
  - Optional: CUDA GPU (falls back to CPU)

Run:
    python examples/benchmark_sweep.py

Expected output:
    Executed benchmark evaluation and printed summary DataFrame.
"""
from neuralops.benchmarks.api import Runner


def main() -> None:
    print("Configuring lightweight evaluation sweep across representative modules...")
    runner = Runner(
        modules=["cosformer", "svd_linear_attention"],
        models="distilbert-base-uncased",
        datasets="sst2",
        training_type="none",  # Evaluation only
        device="cpu",
    )

    print("Executing benchmark stream...")
    for res in runner.stream():
        status = res.status.upper()
        mod_name = res.module.get("name") if isinstance(res.module, dict) else res.module
        print(f"[{status}] Module: {mod_name}")

    df = runner.summary()
    print(f"\nExecution summary ({len(df)} records):")
    if not df.empty:
        print(df.to_string(index=False))

    print("\n✅ Executed benchmark evaluation and printed summary DataFrame.")


if __name__ == "__main__":
    main()
