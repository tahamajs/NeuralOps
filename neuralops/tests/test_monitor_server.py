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

def test_advanced_monitor_endpoints():
    """Verify new 131 Attentions, 72 Compressors, PagedKV, and Speculative Decoding APIs."""
    import os
    from neuralops.runners.monitor.advanced_data import (
        get_all_131_attentions,
        get_all_72_compressors,
        get_paged_kv_cache_state,
        get_speculative_decoding_state,
    )
    from neuralops.runners.monitor.fastapi_app import create_app

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    attns = get_all_131_attentions(project_dir)
    assert len(attns) >= 131, f"Expected at least 131 attentions, got {len(attns)}"
    assert "name" in attns[0]

    comps = get_all_72_compressors(project_dir)
    assert len(comps) >= 72, f"Expected at least 72 compressors, got {len(comps)}"
    assert "rel_err" in comps[0]

    paged = get_paged_kv_cache_state()
    assert "allocated_blocks" in paged
    assert "blocks" in paged
    assert len(paged["blocks"]) == 64

    spec = get_speculative_decoding_state()
    assert spec["acceptance_rate"] > 0.0
    assert spec["speedup_factor"] > 1.0

    app = create_app(project_dir)
    assert app.title == "NeuralOps Research Platform & Telemetry Monitor"
