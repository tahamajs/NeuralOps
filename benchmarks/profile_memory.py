"""Profile memory usage of NeuralOps modules."""
import gc
import tracemalloc

import torch


def measure_memory(func, *args, **kwargs):
    """Measure peak memory of a function."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    tracemalloc.start()
    result = func(*args, **kwargs)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    cuda_peak = 0
    if torch.cuda.is_available():
        cuda_peak = torch.cuda.max_memory_allocated() / 1e6

    return {
        "cpu_peak_mb": peak / 1e6,
        "cuda_peak_mb": cuda_peak,
        "result": result,
    }


if __name__ == "__main__":
    from neuralops.core.registry import ATTENTION_REGISTRY
    import neuralops.attentions  # noqa: F401

    for module_name in ["svd_linear_attention", "cosformer", "gqa_attention"]:
        cls = ATTENTION_REGISTRY.get(module_name)
        if cls is None:
            continue
        try:
            attn = cls(hidden_size=256, num_heads=4)
        except Exception:
            continue

        x = torch.randn(2, 512, 256)
        with torch.no_grad():
            mem = measure_memory(attn, x)
        print(f"{module_name:<30} CPU: {mem['cpu_peak_mb']:>6.1f} MB")
