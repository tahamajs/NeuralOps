Module neuralops.utils.profiler
===============================

Functions
---------

`measure_inference_latency(model, dataloader, device, num_samples: int = 100, warmup: int = 10) ‑> dict[str, float]`
:   Measure inference latency.

`measure_memory_usage(model, dataloader, device, num_samples: int = 10) ‑> dict[str, float]`
:   Measure memory usage.

`profile_context(model, profiler: neuralops.utils.profiler.ModelProfiler | None = None) ‑> None`
:   Profile context.

Classes
-------

`ModelProfiler(model, device=None)`
:   

    ### Methods

    `get_report(self) ‑> dict[str, dict[str, float]]`
    :   Return the report.

    `print_report(self) ‑> None`
    :   Print report.

    `save_report(self, path: str) ‑> None`
    :   Persist report to disk.

    `start_profiling(self) ‑> None`
    :   Start profiling.

    `stop_profiling(self) ‑> None`
    :   Stop profiling.