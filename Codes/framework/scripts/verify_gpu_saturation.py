"""
GPU Utilization & Saturation Monitor.

Samples nvidia-smi every 5 seconds for a configurable duration (default 10 minutes)
and computes time-series metrics, mean GPU utilization, peak VRAM, and percentage
of time utilization stays >= 80%.
"""

import time
import argparse
import subprocess
import json
import statistics
from datetime import datetime
from pathlib import Path


def sample_nvidia_smi():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0 and res.stdout.strip():
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            if len(parts) >= 5:
                def _to_float(v, default=0.0):
                    try:
                        return float(v)
                    except Exception:
                        return default
                return {
                    "timestamp": datetime.now().isoformat(),
                    "gpu_util_pct": _to_float(parts[0]),
                    "mem_used_mb": _to_float(parts[1]),
                    "mem_total_mb": _to_float(parts[2], 8188.0),
                    "power_w": _to_float(parts[3]),
                    "temp_c": _to_float(parts[4]),
                }
    except Exception as e:
        return None


def run_gpu_monitor(duration_sec=600, interval_sec=5, output_file="gpu_saturation_log.json"):
    print(f"Starting GPU Saturation Sampling for {duration_sec}s (interval={interval_sec}s)...", flush=True)
    samples = []
    start_t = time.time()
    
    while time.time() - start_t < duration_sec:
        sample = sample_nvidia_smi()
        if sample:
            samples.append(sample)
            elapsed = time.time() - start_t
            print(f"[{elapsed:5.1f}s / {duration_sec}s] GPU Util: {sample['gpu_util_pct']:5.1f}% | VRAM: {sample['mem_used_mb']:5.0f} MB | Power: {sample['power_w']:5.1f} W | Temp: {sample['temp_c']:3.0f} °C", flush=True)
        time.sleep(interval_sec)
        
    if not samples:
        print("No samples collected.")
        return

    utils = [s["gpu_util_pct"] for s in samples]
    vrams = [s["mem_used_mb"] for s in samples]
    powers = [s["power_w"] for s in samples]
    
    mean_util = statistics.mean(utils)
    median_util = statistics.median(utils)
    pct_above_80 = (sum(1 for u in utils if u >= 80.0) / len(utils)) * 100.0
    pct_above_60 = (sum(1 for u in utils if u >= 60.0) / len(utils)) * 100.0
    
    summary = {
        "duration_sec": duration_sec,
        "total_samples": len(samples),
        "mean_gpu_util_pct": round(mean_util, 2),
        "median_gpu_util_pct": round(median_util, 2),
        "min_gpu_util_pct": round(min(utils), 2),
        "max_gpu_util_pct": round(max(utils), 2),
        "pct_time_ge_80": round(pct_above_80, 2),
        "pct_time_ge_60": round(pct_above_60, 2),
        "mean_vram_mb": round(statistics.mean(vrams), 1),
        "peak_vram_mb": round(max(vrams), 1),
        "mean_power_w": round(statistics.mean(powers), 2),
        "samples": samples,
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n========================================================")
    print("GPU SATURATION MONITOR REPORT")
    print("========================================================")
    print(f"Total Duration:        {duration_sec} s ({len(samples)} samples)")
    print(f"Mean GPU Utilization:  {mean_util:.2f}%")
    print(f"Median GPU Util:       {median_util:.2f}%")
    print(f"Time GPU >= 80%:       {pct_above_80:.2f}%")
    print(f"Time GPU >= 60%:       {pct_above_60:.2f}%")
    print(f"Peak VRAM Allocation:  {max(vrams):.1f} MB")
    print(f"Mean Power Draw:       {statistics.mean(powers):.2f} W")
    print(f"Detailed log saved to: {output_file}")
    print("========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=600, help="Total sampling duration in seconds (default 600s)")
    parser.add_argument("--interval", type=int, default=5, help="Sampling interval in seconds (default 5s)")
    parser.add_argument("--out", type=str, default="gpu_saturation_log.json", help="Output json path")
    args = parser.parse_args()
    run_gpu_monitor(duration_sec=args.duration, interval_sec=args.interval, output_file=args.out)
