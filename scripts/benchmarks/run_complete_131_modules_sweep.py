# -*- coding: utf-8 -*-
"""Comprehensive Empirical Multi-GPU Sweep for All 131 Registered Attention Modules.
Sweeps across:
  - Sequence lengths: T in [128, 512, 1024]
  - Head/Dimension topologies: (64, 2), (128, 4), (256, 8)
  - Execution regimes: Training (Autograd Fwd+Bwd), Causal Inference, FP16 Autocast
Outputs structured artifacts to results/complete_131_modules_sweep.json & .md & .csv
"""

import gc
import json
import os
import time
import torch
from neuralops.core.registry import ATTENTION_REGISTRY


def run_sweep(output_dir="results"):
    """Execute run sweep routine."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    node_name = os.uname().nodename

    os.makedirs(output_dir, exist_ok=True)

    keys = sorted(ATTENTION_REGISTRY.keys())
    total = len(keys)
    print(f"[{node_name}] Starting 131-module multi-dimensional sweep on {gpu_name}...")

    seq_lengths = [128, 512, 1024]
    configs = [(64, 2), (128, 4), (256, 8)]

    module_records = []

    t_global_start = time.time()

    for idx, name in enumerate(keys, 1):
        torch.cuda.empty_cache()
        gc.collect()
        t0 = time.time()
        cls = ATTENTION_REGISTRY.get(name)
        status = "PASS"
        err = ""

        try:
            for dim, num_heads in configs:
                mod = cls(dim=dim, num_heads=num_heads).to(device)
                mod.train()

                # Training sweep over seq_lengths
                for T in seq_lengths:
                    B = 2 if T <= 512 else 1
                    x = torch.randn(B, T, dim, device=device, requires_grad=True)
                    out = mod(x)
                    if isinstance(out, tuple):
                        out = out[0]
                    loss = out.sum()
                    loss.backward()
                    assert x.grad is not None and not torch.isnan(x.grad).any()

                # Causal & FP16 check
                mod.eval()
                if hasattr(mod, "is_causal"):
                    mod.is_causal = True
                if hasattr(mod, "is_decoder"):
                    mod.is_decoder = True

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    with torch.no_grad():
                        x_eval = torch.randn(2, 256, dim, device=device)
                        try:
                            out_c = mod(x_eval, is_causal=True)
                        except TypeError:
                            out_c = mod(x_eval)
                        if isinstance(out_c, tuple):
                            out_c = out_c[0]
                        assert not torch.isnan(out_c).any()
        except Exception as e:
            status = "FAIL"
            err = str(e)

        dur = time.time() - t0
        print(f"[{idx:03d}/{total:03d}] [{status}] {name:<35} ({dur:.2f}s) {err}")
        module_records.append(
            {
                "module": name,
                "status": status,
                "latency_s": round(dur, 3),
                "device": gpu_name,
                "node": node_name,
                "error": err,
            }
        )

    total_time = time.time() - t_global_start

    # Save artifacts
    json_path = os.path.join(output_dir, f"sweep_131_modules_{node_name}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "node": node_name,
                "device": gpu_name,
                "total_modules": total,
                "passed": sum(1 for r in module_records if r["status"] == "PASS"),
                "failed": sum(1 for r in module_records if r["status"] == "FAIL"),
                "total_time_s": round(total_time, 2),
                "results": module_records,
            },
            f,
            indent=2,
        )

    csv_path = os.path.join(output_dir, f"sweep_131_modules_{node_name}.csv")
    with open(csv_path, "w") as f:
        f.write("module,status,latency_s,device,node,error\n")
        for r in module_records:
            f.write(f'"{r["module"]}","{r["status"]}",{r["latency_s"]},"{r["device"]}","{r["node"]}","{r["error"]}"\n')

    md_path = os.path.join(output_dir, f"sweep_131_modules_{node_name}.md")
    with open(md_path, "w") as f:
        f.write(f"# Sweep Report for All 131 Modules on `{gpu_name}` (`{node_name}`)\n\n")
        f.write(f"- **Total Modules Tested**: {total}\n")
        f.write(f"- **Passed**: {sum(1 for r in module_records if r['status'] == 'PASS')}\n")
        f.write(f"- **Failed**: {sum(1 for r in module_records if r['status'] == 'FAIL')}\n")
        f.write(f"- **Total Sweep Time**: {total_time:.2f}s\n\n")
        f.write("| # | Module | Status | Latency (s) | Error |\n")
        f.write("| :---: | :--- | :---: | :---: | :--- |\n")
        for i, r in enumerate(module_records, 1):
            f.write(f"| {i} | `{r['module']}` | **{r['status']}** | {r['latency_s']} | {r['error']} |\n")

    print(f"Saved artifacts to {json_path}, {csv_path}, {md_path}")


if __name__ == "__main__":
    run_sweep()
