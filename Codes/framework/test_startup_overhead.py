"""
Benchmark & Verify Startup Overhead and GPU Utilization Across Modules.
"""

import sys
import time
import json
import logging
from pathlib import Path

_HERE = Path(__file__).resolve()
_CODES_DIR = str(_HERE.parent.parent)
_PROJ_DIR = str(_HERE.parent.parent.parent)
for _p in [_PROJ_DIR, _CODES_DIR, str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OverheadBenchmark")

def run_overhead_test(modules=("full", "cosformer", "svd_linear"), batch_size=128, seq_len=128, steps=20):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running startup overhead test on {device}...")
    
    from framework.persistent_worker import PersistentCUDAWorker
    
    worker = PersistentCUDAWorker(
        device_str=str(device),
        pretrained=True,
        model_family="distilbert",
        dataset="sst2",
        seq_len=seq_len,
    )
    
    results = []
    logger.info("================================================================")
    logger.info("WARMING UP & RUNNING PERSISTENT WORKER EVALUATION")
    logger.info("================================================================")
    
    for mod in modules:
        t0 = time.time()
        res, startup_s = worker.execute_task(
            module_name=mod,
            model_family="distilbert",
            dataset="sst2",
            seed=42,
            epochs=1,
            max_steps=steps,
            batch_size=batch_size,
            seq_len=seq_len,
            pretrained=True,
            timeout=120,
        )
        total_wall_s = time.time() - t0
        logger.info(
            f"--> Module {mod:15s} | Startup Overhead: {startup_s*1000:6.1f} ms | Total Time: {total_wall_s:5.2f}s | Acc: {res.accuracy_pct:.2f}% | F1: {res.f1_score:.4f}"
        )
        results.append({
            "module": mod,
            "startup_overhead_ms": round(startup_s * 1000.0, 2),
            "total_time_s": round(total_wall_s, 2),
            "accuracy_pct": res.accuracy_pct,
            "f1_score": res.f1_score,
            "peak_vram_mb": res.peak_vram_mb,
        })
        
    worker.close()
    
    avg_startup = sum(r["startup_overhead_ms"] for r in results) / len(results)
    logger.info("================================================================")
    logger.info(f"AVERAGE PER-MODULE STARTUP OVERHEAD: {avg_startup:.1f} ms (< 2000 ms target)")
    logger.info("================================================================")
    return results

if __name__ == "__main__":
    run_overhead_test()
