"""Benchmark attention modules (CPU only)."""
import time
import json
from neuralops.core.registry import ATTENTION_REGISTRY
import neuralops.attentions
import torch

def benchmark_module(name, B=2, N=128, D=64, H=4):
    cls = ATTENTION_REGISTRY.get(name)
    if cls is None:
        return {"module": name, "status": "not_found"}
    try:
        attn = cls(hidden_size=H*D, num_heads=H)
    except Exception as e:
        return {"module": name, "status": "init_failed", "error": str(e)[:100]}
    
    x = torch.randn(B, N, H*D)
    
    # Warmup
    try:
        with torch.no_grad():
            for _ in range(2):
                attn(x)
    except Exception as e:
        return {"module": name, "status": "forward_failed", "error": str(e)[:100]}
    
    # Benchmark
    start = time.perf_counter()
    try:
        with torch.no_grad():
            for _ in range(5):
                attn(x)
        elapsed = (time.perf_counter() - start) / 5
        return {"module": name, "status": "ok", "time_ms": elapsed * 1000}
    except Exception as e:
        return {"module": name, "status": "benchmark_failed", "error": str(e)[:100]}

if __name__ == "__main__":
    results = []
    ok = 0
    failed = 0
    # First 30 attention mechanisms
    for name in sorted(ATTENTION_REGISTRY.keys())[:30]:
        result = benchmark_module(name)
        results.append(result)
        status = result["status"]
        if status == "ok":
            ok += 1
            print(f"OK    {name}: {result['time_ms']:.2f} ms")
        else:
            failed += 1
            print(f"{status:20s} {name}")
    
    with open("benchmarks/attention_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTotal: {ok} OK, {failed} failed")
