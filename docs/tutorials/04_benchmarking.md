# Tutorial 4: Subquadratic Attention Benchmarking

Benchmark latency, peak memory (VRAM), and throughput across sequence lengths.

```python
import torch
import time
from neuralops.core.registry import ATTENTION_REGISTRY
import neuralops.attentions

attn = ATTENTION_REGISTRY["rpc_attention"]()

# Benchmark sequence scaling (N = 512, 1024, 2048)
for seq_len in [512, 1024, 2048]:
    x = torch.randn(1, seq_len, 64)
    start = time.perf_counter()
    for _ in range(10):
        _ = attn(x)
    elapsed_ms = (time.perf_counter() - start) / 10 * 1000
    print(f"Sequence Length {seq_len:4d}: {elapsed_ms:.2f} ms")
```
