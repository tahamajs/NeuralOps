import time
import psutil
import torch
from contextlib import contextmanager
from collections import defaultdict
import json

class ModelProfiler:
    """Profile memory, time, and FLOPs per layer."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        
    def start_profiling(self):
        def hook_fn(module, input, output, name):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            yield
            end_time = time.perf_counter()
            self.timings[name].append(end_time - start_time)
            if torch.cuda.is_available():
                self.memory_usage[name].append(torch.cuda.memory_allocated() / 1024**2)
        
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'ffn' in name.lower():
                hook = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
                self.hooks.append(hook)
    
    def stop_profiling(self):
        for hook in self.hooks:
            hook.remove()
    
    def get_report(self):
        report = {}
        for name in self.timings:
            report[name] = {
                'avg_time_ms': 1000 * np.mean(self.timings[name]),
                'std_time_ms': 1000 * np.std(self.timings[name]),
                'avg_memory_mb': np.mean(self.memory_usage[name]) if self.memory_usage[name] else 0,
            }
        return report
    
    def save_report(self, path):
        with open(path, 'w') as f:
            json.dump(self.get_report(), f, indent=2)


def measure_inference_latency(model, dataloader, device, num_samples=100):
    """Measure inference latency (ms per sample)."""
    model.eval()
    latencies = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            start = time.perf_counter()
            _ = model(**batch)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
    }