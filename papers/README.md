# GPU Benchmark Execution Artifacts (`papers/`)

This directory stores intermediate run outputs, power consumption metrics, and rank-ablation sweep checkpoints generated during automated GPU test runs:

- **`modules/`**: Benchmark runs, per-model cards, and logs for individual modules.
- **`_rank_ablation/`**: Rank sensitivity sweeps ($r \in \{16, 32, 64, 128, 256\}$).
- **`_baseline_power.json` & `_optimized_power.json`**: GPU wattage and hardware energy profiling data.

> [!NOTE]
> All LaTeX manuscripts, camera-ready PDFs, and conference submission files are located in [`../paper/`](../paper/).
