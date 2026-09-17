# Resource Accounting

## Purpose

Report computational and environmental cost.

## Metrics

- Total GPU-hours
- Estimated CO2 (kg CO2eq)
- Energy per inference (joules)
- FLOPs (theoretical and measured)
- Kernel launch counts
- Peak memory bandwidth

## Implementation

```python
import time, torch

def measure_inference(model, batch, n=20):
    model.eval()
    latencies = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(**batch)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - t0)
    return latencies

def measure_flops(model, batch):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            model(**batch)
    return sum(e.flops for e in prof.key_averages())
```

## CO2 estimation

```python
def estimate_co2(gpu_hours, gpu_tdp_watts=115, pue=1.1, carbon_intensity_g_per_kwh=475):
    energy_kwh = gpu_hours * gpu_tdp_watts / 1000 * pue
    return energy_kwh * carbon_intensity_g_per_kwh / 1000  # kg
```

## Output

- `papers/modules/<module>/metrics/compute.json`
- `papers/modules/<module>/metrics/carbon.json`
- `papers/modules/<module>/metrics/energy_per_inference.json`
- `papers/modules/<module>/metrics/flops.json`
- `papers/modules/<module>/metrics/measured_flops.json`
- `papers/modules/<module>/metrics/kernel_counts.json`
