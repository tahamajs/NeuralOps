# framework/utils/profiler.py
"""
Model profiling utilities for memory, time, and FLOPs.
"""

import time
import json
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, Any, Optional
from contextlib import contextmanager


class ModelProfiler:
    """Profile memory, time, and FLOPs per layer."""
    
    def __init__(self, model, device=None):
        """
        Args:
            model: PyTorch model to profile
            device: Device to run on
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.hooks = []
        self.timings = defaultdict(list)
        self.memory_usage = defaultdict(list)
        self._current_timings = {}
        
    def _get_layer_name(self, name: str) -> str:
        """Extract meaningful layer name."""
        parts = name.split('.')
        # Try to identify layer type
        for part in parts:
            if part in ['attention', 'self', 'output', 'intermediate', 'ffn', 'mlp']:
                return '.'.join([p for p in parts if p.isdigit() or p in ['attention', 'ffn', 'mlp']])
        return name
    
    def start_profiling(self):
        """Start profiling by registering hooks."""
        
        def pre_hook_fn(module, input, name):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self._current_timings[name] = time.perf_counter()
            if torch.cuda.is_available():
                self.memory_usage[name].append(torch.cuda.memory_allocated() / 1024**2)
        
        def post_hook_fn(module, input, output, name):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            if name in self._current_timings:
                elapsed = end_time - self._current_timings[name]
                self.timings[name].append(elapsed * 1000)  # Convert to ms
                del self._current_timings[name]
        
        for name, module in self.model.named_modules():
            # Profile attention and FFN layers
            if any(layer_type in name.lower() for layer_type in ['attention', 'ffn', 'mlp', 'intermediate']):
                
                def make_pre_hook(n):
                    return lambda m, i: pre_hook_fn(m, i, n)
                
                def make_post_hook(n):
                    return lambda m, i, o: post_hook_fn(m, i, o, n)
                
                pre_hook = module.register_forward_pre_hook(make_pre_hook(name))
                post_hook = module.register_forward_hook(make_post_hook(name))
                self.hooks.extend([pre_hook, post_hook])
    
    def stop_profiling(self):
        """Stop profiling and remove hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_report(self) -> Dict[str, Dict[str, float]]:
        """
        Get profiling report.
        
        Returns:
            Dictionary with timing and memory statistics per layer
        """
        report = {}
        for name in self.timings:
            times = self.timings[name]
            mems = self.memory_usage[name] if self.memory_usage[name] else [0]
            
            report[self._get_layer_name(name)] = {
                'avg_time_ms': np.mean(times) if times else 0,
                'std_time_ms': np.std(times) if times else 0,
                'min_time_ms': np.min(times) if times else 0,
                'max_time_ms': np.max(times) if times else 0,
                'avg_memory_mb': np.mean(mems),
            }
        
        # Add total statistics
        if self.timings:
            all_times = []
            for times in self.timings.values():
                all_times.extend(times)
            report['_total'] = {
                'avg_time_ms': np.mean(all_times) if all_times else 0,
                'std_time_ms': np.std(all_times) if all_times else 0,
            }
        
        return report
    
    def save_report(self, path: str):
        """Save profiling report to JSON file."""
        report = self.get_report()
        with open(path, 'w') as f:
            json.dump(report, f, indent=2)
    
    def print_report(self):
        """Print profiling report to console."""
        report = self.get_report()
        print("\n" + "=" * 60)
        print("PROFILING REPORT")
        print("=" * 60)
        
        for name, stats in report.items():
            if name == '_total':
                print(f"\n📊 TOTAL: {stats['avg_time_ms']:.2f}ms (±{stats['std_time_ms']:.2f})")
            else:
                print(f"\n🔹 {name}:")
                print(f"   Time: {stats['avg_time_ms']:.2f}ms (±{stats['std_time_ms']:.2f})")
                print(f"   Memory: {stats['avg_memory_mb']:.2f} MB")


def measure_inference_latency(
    model,
    dataloader,
    device,
    num_samples: int = 100,
    warmup: int = 10
) -> Dict[str, float]:
    """
    Measure inference latency (ms per sample).
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to measure
        warmup: Number of warmup runs
        
    Returns:
        Dictionary with latency statistics
    """
    model.eval()
    model.to(device)
    
    latencies = []
    
    # Warmup
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _ = model(**batch)
    
    # Measure
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(**batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # ms
    
    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p95_latency_ms': np.percentile(latencies, 95),
        'p99_latency_ms': np.percentile(latencies, 99),
        'samples_per_second': 1000 / np.mean(latencies) if np.mean(latencies) > 0 else 0,
    }


def measure_memory_usage(model, dataloader, device, num_samples: int = 10) -> Dict[str, float]:
    """
    Measure peak memory usage during inference.
    
    Args:
        model: PyTorch model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to measure
        
    Returns:
        Dictionary with memory statistics
    """
    model.eval()
    model.to(device)
    
    peak_memory = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
            
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            _ = model(**batch)
            
            if torch.cuda.is_available():
                peak_memory.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
    
    return {
        'peak_memory_mb': np.max(peak_memory) if peak_memory else 0,
        'avg_peak_memory_mb': np.mean(peak_memory) if peak_memory else 0,
    }


@contextmanager
def profile_context(model, profiler: Optional[ModelProfiler] = None):
    """Context manager for profiling."""
    if profiler is None:
        profiler = ModelProfiler(model)
    profiler.start_profiling()
    yield profiler
    profiler.stop_profiling()