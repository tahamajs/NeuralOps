"""Unit tests for neuralops.runners.monitor package."""

import json
from neuralops.runners.monitor.telemetry import get_live_gpu_telemetry, get_live_logs
from neuralops.runners.monitor.rendering import render_favicon, render_csv_export, render_json_export
from neuralops.runners.monitor.endpoints import get_loss_history, get_doc_content, stop_current_run, launch_new_run
from neuralops.runners.monitor.stats import get_module_doc_index


def test_gpu_telemetry_fallback():
    gpu = get_live_gpu_telemetry("/nonexistent")
    assert "gpu_util" in gpu
    assert "mem_used_mb" in gpu
    assert gpu["mem_total_mb"] > 0


def test_live_logs_fallback():
    logs = get_live_logs("/nonexistent")
    assert "lines" in logs
    assert logs["count"] >= 1


def test_render_favicon():
    svg = render_favicon()
    assert b"<svg" in svg


def test_render_csv_and_json_export():
    records = [{"module": "svd_attn", "acc": 88.5}]
    csv_bytes = render_csv_export(records)
    assert b"svd_attn" in csv_bytes

    stats = {"status": "ok", "count": 1}
    json_bytes = render_json_export(stats)
    assert b"\"count\": 1" in json_bytes


def test_endpoints():
    loss = get_loss_history("/nonexistent")
    assert loss["success"] is True

    doc = get_doc_content("/nonexistent", "/nonexistent", "svd_linear_attention")
    assert doc["success"] is True
    assert "content" in doc

    launch = launch_new_run({"rank": 16})
    assert launch["success"] is True
    assert launch["params"]["rank"] == 16
