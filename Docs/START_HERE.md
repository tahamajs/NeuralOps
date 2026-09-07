# SVD Linear Attention Documentation

This is the canonical map for the `Docs/` knowledge base. The [Research Hub](./index.html) loads the same categories from [`catalog.json`](./catalog.json), so every guide, experiment report, architecture, and studio specification has one discoverable entry point.

For the broad end-to-end explanation, read the [Complete Research Guide](./COMPLETE_RESEARCH_GUIDE.md). It connects architecture theory, framework code, verification, experiments, results, studios, and deployment.

## Choose a path

| Goal | Start here | What follows |
| --- | --- | --- |
| Understand the project | [Attention survey](./Attention_Mechanisms_For_Paper.md) | Architecture notes and mathematical background |
| Run the framework | [Deployment guide](./DEPLOYMENT.md) and [Kaggle/GPU guide](./DEPLOYMENT.md) | Server, GPU, telemetry, and Pages setup |
| Explore architectures | [Attention catalog](./content/attentions/) | Attention mechanisms, hybrids, and SSM variants |
| Explore compression | [Compression catalog](./content/compressions/) | SVD, quantization, pruning, and low-rank engines |
| Read research deep dives | [Attention series](./blog/01_spectral_svd_lowrank_attentions.md) | Five linked series covering attention and compression |
| Inspect evidence | [Verification reports](./reports/) and [results](./results/) | Reproducibility checks, metrics, summaries, and Pareto data |
| Use interactive studies | [Attention visualization dashboard](./Attention_Visualization_Dashboard/Attention_Visualization_Dashboard.md) | Canvas, math, analytics, and transformer-component studios |
| Understand internals | [Framework docs](./content/framework/) | Registries, training, distributed execution, and utilities |

## Documentation contract

- Architecture specifications describe what a module does and link to its implementation under `Codes/framework/`.
- Verification reports describe how behavior was checked; they do not replace benchmark results.
- Benchmark reports and JSON/CSV files under `results/` are evidence consumed by the UI leaderboard and experiment workflows.
- Studio specifications describe interactive surfaces and link back to the relevant architecture or experiment concepts.
- Run `python generate_catalog.py` after adding or renaming Markdown files. The CI workflow verifies that the generated catalog is committed.

## Local preview

```bash
python generate_catalog.py
python -m http.server 8080 --directory Docs
```

Then open <http://127.0.0.1:8080/> and use the search palette or sidebar sections to move between related material.
