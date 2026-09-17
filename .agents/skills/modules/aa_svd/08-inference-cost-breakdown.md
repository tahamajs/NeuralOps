# Inference Cost Breakdown — aa_svd

## Purpose

Explain why inference is faster/slower than baseline. This is required in the
paper's Analysis section.

## Metrics to collect

| Metric | Baseline | aa_svd | Ratio |
|---|---|---|---|
| Latency (s) | ? | ? | ? |
| Kernel launches | ? | ? | ? |
| Peak VRAM (MB) | ? | ? | ? |
| Python overhead (ms) | ? | ? | ? |
| FP16→FP32 casts | ? | ? | ? |
| Sequential matmuls | ? | ? | ? |

## Method

```python
from torch.profiler import profile, ProfilerActivity

def profile_inference(model, batch):
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        with torch.no_grad():
            model(**batch)
    return {
        "cuda_time_ms": sum(e.cuda_time_total for e in prof.key_averages()) / 1000,
        "cpu_time_ms": sum(e.cpu_time_total for e in prof.key_averages()) / 1000,
        "kernel_count": len([e for e in prof.key_averages() if e.device_type == "cuda"]),
    }
```

## Expected for aa_svd

<what the module is expected to cost>

## Output

`papers/modules/aa_svd/metrics/inference_cost.json`
