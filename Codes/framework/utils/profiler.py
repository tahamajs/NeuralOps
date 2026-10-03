# framework/utils/profiler.py
import json
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


class ModelProfiler:
    def __init__(self, model, device=None) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.hooks = []
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self._current_timings = {}

    def _get_layer_name(self, name: str) -> str:
        """get layer name."""
        parts = name.split(".")
        filtered = [
            p
            for p in parts
            if p.isdigit() or p in ("attention", "ffn", "mlp", "intermediate")
        ]
        return ".".join(filtered) if filtered else name

    def start_profiling(self) -> None:
        """Start profiling."""

        def pre_hook_fn(module, input, name) -> None:
            """Pre hook fn."""
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._current_timings[name] = time.perf_counter()
            if torch.cuda.is_available():
                self.memory_usage[name].append(torch.cuda.memory_allocated() / 1024**2)

        def post_hook_fn(module, input, output, name) -> None:
            """Post hook fn."""
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if name in self._current_timings:
                elapsed = time.perf_counter() - self._current_timings[name]
                self.timings[name].append(elapsed * 1000)
                del self._current_timings[name]

        for name, module in self.model.named_modules():
            if any(
                layer in name.lower()
                for layer in ("attention", "ffn", "mlp", "intermediate")
            ):
                pre = module.register_forward_pre_hook(
                    lambda m, i, n=name: pre_hook_fn(m, i, n)
                )
                post = module.register_forward_hook(
                    lambda m, i, o, n=name: post_hook_fn(m, i, o, n)
                )
                self.hooks.extend([pre, post])

    def stop_profiling(self) -> None:
        """Stop profiling."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_report(self) -> dict[str, dict[str, float]]:
        """Return the report."""
        report = {}
        for name in self.timings:
            times = self.timings[name]
            mems = self.memory_usage[name] if self.memory_usage[name] else [0]
            report[self._get_layer_name(name)] = {
                "avg_time_ms": np.mean(times) if times else 0,
                "std_time_ms": np.std(times) if times else 0,
                "min_time_ms": np.min(times) if times else 0,
                "max_time_ms": np.max(times) if times else 0,
                "avg_memory_mb": np.mean(mems),
            }
        if self.timings:
            all_times = [t for ts in self.timings.values() for t in ts]
            report["_total"] = {
                "avg_time_ms": np.mean(all_times) if all_times else 0,
                "std_time_ms": np.std(all_times) if all_times else 0,
            }
        return report

    def save_report(self, path: str) -> None:
        """Persist report to disk."""
        with open(path, "w") as f:
            json.dump(self.get_report(), f, indent=2)

    def print_report(self) -> None:
        """Print report."""
        report = self.get_report()
        print("\n" + "=" * 60)
        print("PROFILING REPORT")
        print("=" * 60)
        for name, stats in report.items():
            if name == "_total":
                print(
                    f"\n📊 TOTAL: {stats['avg_time_ms']:.2f}ms (±{stats['std_time_ms']:.2f})"
                )
            else:
                print(f"\n🔹 {name}:")
                print(
                    f"   Time: {stats['avg_time_ms']:.2f}ms (±{stats['std_time_ms']:.2f})"
                )
                if "avg_memory_mb" in stats:
                    print(f"   Memory: {stats['avg_memory_mb']:.2f} MB")


def measure_inference_latency(
    model, dataloader, device, num_samples: int = 100, warmup: int = 10
) -> dict[str, float]:
    """Measure inference latency."""
    model.eval()
    model.to(device)
    latencies = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup + num_samples:
                break
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(**batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            if i >= warmup:
                latencies.append((end - start) * 1000)
    if not latencies:
        return {}
    return {
        "mean_latency_ms": np.mean(latencies),
        "std_latency_ms": np.std(latencies),
        "min_latency_ms": np.min(latencies),
        "max_latency_ms": np.max(latencies),
        "p50_latency_ms": np.percentile(latencies, 50),
        "p95_latency_ms": np.percentile(latencies, 95),
        "p99_latency_ms": np.percentile(latencies, 99),
        "samples_per_second": (
            1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0
        ),
    }


def measure_memory_usage(
    model, dataloader, device, num_samples: int = 10
) -> dict[str, float]:
    """Measure memory usage."""
    model.eval()
    model.to(device)
    peak_memory = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            batch = {
                k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            _ = model(**batch)
            if torch.cuda.is_available():
                peak_memory.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
    return {
        "peak_memory_mb": np.max(peak_memory) if peak_memory else 0,
        "avg_peak_memory_mb": np.mean(peak_memory) if peak_memory else 0,
    }


@contextmanager
def profile_context(model, profiler: ModelProfiler | None = None) -> None:
    """Profile context."""
    if profiler is None:
        profiler = ModelProfiler(model)
    profiler.start_profiling()
    yield profiler
    profiler.stop_profiling()
