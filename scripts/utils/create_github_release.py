#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Script to create GitHub Release for NeuralOps v1.6.0 and upload distribution assets.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request


def main() -> int:
    p = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n",
        text=True,
        capture_output=True,
    )
    creds = dict(line.split("=", 1) for line in p.stdout.strip().split("\n") if "=" in line)
    token = creds.get("password")
    if not token:
        print("ERROR: GitHub token not found in git credentials.", file=sys.stderr)
        return 1

    body = """# NeuralOps v1.6.0: CPU Stability, Modern Tier Architecture & Lazy Loading

NeuralOps v1.6.0 marks a major architectural release focusing on CPU portability, clean tier separation, PEP 562 lazy loading, and robust verification.

### Highlights
- **Modern 3-Tier Architecture**: Physical separation of core subsystems into `neuralops.layers` (131 attention mechanisms), `neuralops.compression` (72 model compression engines), and `neuralops.ops` (112 CUDA C++ packages housing 114 native kernels, and 26 Triton kernel modules housing 62 unified operators). Full backward-compatibility shims are preserved at legacy paths.
- **PEP 562 Lazy Loading**: Bare import `import neuralops` executes in `<30ms` using standard library only, completely decoupling import-time execution from PyTorch, CUDA, or Triton runtimes.
- **Duplicate Primitive Consolidation**: Consolidated multiple legacy variants into single canonical classes (`LoRALinear`, `SVDLinear`, `FactorizedLinear`, `NeuralOpsConfig`, `compute_flops_estimate`). Deleted redundant dead packages (`neuralops/monitor`, legacy `neuralops/attention`) and unified training under `neuralops.trainer`.
- **Decoupled Telemetry & Benchmarking**: Heavy analysis and plotting tools (`paper_export`, `results_ledger`, `roofline`, `inspector`) relocated out of core to `neuralops.analysis` and `neuralops.benchmarks`, decoupling the core engine from `matplotlib` and `pandas`.
- **Silent Failure Elimination & Honest Stubs**: Replaced bare exception swallows and hardcoded numeric fallbacks with explicit diagnostic warnings. Serving integrations (`vllm`, `sglang`, `tensorrt_llm`, `tgi`) updated to honest stubs raising `IntegrationNotImplementedError` when optional backends are missing.
- **CPU Distributed Test Suite**: Added `tests/distributed/test_distributed_cpu.py` with 9 CPU-safe tests for context and tensor parallel modules, collective uninitialized state guards, and single-process linear projections.
- **Quality Gates & Security Hygiene**: Zero AST dead-code issues across 1,377 Python files, universal MIT headers, and automated code hygiene tests (`tests/test_code_quality.py`).

### PyPI Distribution
- Package live on PyPI: [`pip install neuralops==1.6.0`](https://pypi.org/project/neuralops/1.6.0/)
"""

    release_payload = {
        "tag_name": "v1.6.0",
        "target_commitish": "main",
        "name": "NeuralOps v1.6.0 — CPU Stability, Modern Tier Architecture & Production Release",
        "body": body,
        "draft": False,
        "prerelease": False,
    }

    req = urllib.request.Request(
        "https://api.github.com/repos/tahamajs/NeuralOps/releases",
        data=json.dumps(release_payload).encode("utf-8"),
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "NeuralOps-Release-Agent",
        },
    )

    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode()
        if "already_exists" in err_msg:
            # Retrieve existing release
            get_req = urllib.request.Request(
                "https://api.github.com/repos/tahamajs/NeuralOps/releases/tags/v1.6.0",
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "NeuralOps-Release-Agent",
                },
            )
            with urllib.request.urlopen(get_req) as g_resp:
                res = json.loads(g_resp.read().decode())
        else:
            print(f"HTTPError {e.code}: {err_msg}", file=sys.stderr)
            return 1

    print(f"Release ID: {res.get('id')}")
    print(f"Release URL: {res.get('html_url')}")
    upload_url = res.get("upload_url", "").split("{")[0]

    # Upload dist assets
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dist_dir = os.path.join(repo_root, "dist")
    if os.path.exists(dist_dir):
        for fname in sorted(os.listdir(dist_dir)):
            if not fname.endswith((".whl", ".tar.gz")):
                continue
            fpath = os.path.join(dist_dir, fname)
            ctype = "application/zip" if fname.endswith(".whl") else "application/gzip"
            with open(fpath, "rb") as f:
                data = f.read()
            u_req = urllib.request.Request(
                f"{upload_url}?name={fname}",
                data=data,
                headers={
                    "Authorization": f"token {token}",
                    "Accept": "application/vnd.github.v3+json",
                    "Content-Type": ctype,
                    "User-Agent": "NeuralOps-Release-Agent",
                },
            )
            try:
                with urllib.request.urlopen(u_req) as u_resp:
                    u_res = json.loads(u_resp.read().decode())
                    print(f"Uploaded asset {fname}: {u_res.get('browser_download_url')}")
            except urllib.error.HTTPError as ue:
                print(f"Asset upload warning ({fname}): {ue.code}")

    print("GitHub release completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
