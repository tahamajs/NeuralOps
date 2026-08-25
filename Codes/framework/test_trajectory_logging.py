#!/usr/bin/env python3
"""
Verification script for step-by-step metric trajectory logging.
Asserts that loss, grad_norm, fwd_latency, and bwd_latency trajectories
are accurately captured across all steps for conference-grade plotting.
"""

import os
import sys
import json
import torch
from pathlib import Path

# Ensure framework is in python path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
codes_dir = current_dir.parent

if str(codes_dir) not in sys.path:
    sys.path.insert(0, str(codes_dir))

from framework.eval_all_models_datasets import evaluate_single_combination_opt
from framework.results_saver.saver import save_result, save_result_json

def test_trajectory_capture():
    print("=" * 70)
    print("Testing Fine-Grained Trajectory Logging (5 Steps on CPU)...")
    print("=" * 70)

    device = torch.device("cpu")
    num_steps = 5
    num_warmup = 1

    # Run evaluation on tiny model configuration
    result = evaluate_single_combination_opt(
        track="track1_svd",
        module_name="svd_linear_attention",
        module_type="attention",
        model_family="distilbert",
        dataset="sst2",
        rank=16,
        num_warmup=num_warmup,
        num_steps=num_steps,
        batch_size=4,
        seq_len=32,
        device=device,
        allow_synthetic=True,
    )

    result_dict = result.to_dict()

    print(f"\n[+] Execution Status: {result.status}")
    print(f"[+] Average Fwd Latency: {result.forward_latency_ms} ms")
    print(f"[+] Average Bwd Latency: {result.backward_latency_ms} ms")
    print(f"[+] Mean Loss: {result.loss}")
    print(f"[+] Mean Grad Norm: {result.grad_norm}")

    # Assertions for Trajectories
    assert "trajectories" in result_dict, "FAIL: 'trajectories' key missing from result dictionary!"
    trajectories = result_dict["trajectories"]

    required_keys = ["loss", "grad_norm", "fwd_latency_ms", "bwd_latency_ms"]
    for k in required_keys:
        assert k in trajectories, f"FAIL: Trajectory key '{k}' missing!"
        values = trajectories[k]
        assert len(values) == num_steps, f"FAIL: Trajectory '{k}' expected length {num_steps}, got {len(values)}!"
        print(f"[✓] {k} Trajectory ({len(values)} steps): {values}")

    # Verify JSON persistence
    test_dir = current_dir / "test_output"
    test_dir.mkdir(exist_ok=True)
    json_path = save_result_json(result, results_dir=test_dir, filename="test_traj.json")
    with open(json_path, "r", encoding="utf-8") as f:
        loaded_json = json.load(f)
    assert "trajectories" in loaded_json, "FAIL: 'trajectories' missing from saved JSON!"
    print(f"\n[✓] JSON Saved Successfully: {json_path}")
    print(f"[✓] JSON Trajectories Verified: {list(loaded_json['trajectories'].keys())}")

    # Verify CSV persistence (ensuring it stays clean without trajectory bloat)
    csv_path = save_result(result, results_dir=test_dir, csv_filename="test_clean.csv")
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_header = f.readline().strip().split(",")
    assert "trajectories" not in csv_header, "FAIL: CSV header contains 'trajectories' (must be excluded)!"
    print(f"[✓] CSV Saved Cleanly (without bloat): {csv_path}")

    # Cleanup test output
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("ALL TRAJECTORY LOGGING & SERIALIZATION ASSERTIONS PASSED [5/5]!")
    print("=" * 70)

if __name__ == "__main__":
    test_trajectory_capture()
