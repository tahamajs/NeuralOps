# Pareto Benchmark Results

| Method | Category | Prefill 2k (ms) | Prefill 4k (ms) | Decode (tok/s) | Peak VRAM (MB) | Recall (%) |
|---|---|---|---|---|---|---|
| Softmax Attention (Dense) | Full Attention | 0.78 | 2.01 | 7167.7 | 61.8 | 51.0% |
| Cosformer (Linear) | Linear Attention | 10.02 | 24.93 | 1262.9 | 487.3 | 46.7% |
| GLA (Gated Linear) | Linear Attention | 0.65 | 1.47 | 2843.4 | 98.1 | 48.4% |
| SVD Linear Attention | Compression | 0.75 | 1.49 | 6150.4 | 61.6 | 47.3% |