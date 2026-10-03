# Experiment results

These files are the machine-readable evidence layer for the Research Hub. The UI uses `all_histories.json` for optional live training plots; the benchmark summaries and Pareto frontier files support reproducible comparisons.

| Artifact | Purpose |
| --- | --- |
| [`master_benchmark_summary.csv`](./master_benchmark_summary.csv) | Tabular benchmark export |
| [`master_benchmark_summary.json`](./master_benchmark_summary.json) | Structured benchmark records |
| [`master_empirical_benchmark_report.md`](./master_empirical_benchmark_report.md) | Human-readable result interpretation |
| [`pareto_frontier.json`](./pareto_frontier.json) | Accuracy/efficiency trade-off data |
| [`all_histories.json`](./all_histories.json) | Optional training histories for telemetry charts |

Results are snapshots, not live guarantees. Record the model, task, device, precision, seed, and software revision before comparing runs.
