import pytest
from pathlib import Path
from unittest.mock import MagicMock
from neuralops.benchmarks.kd.zoo_orchestrator import ZooOrchestrator


def test_zoo_orchestrator_uploader_compat(tmp_path: Path):
    mock_uploader = MagicMock()
    mock_uploader.submit_upload = MagicMock()

    orchestrator = ZooOrchestrator(
        zoo_type="attention",
        out_dir=tmp_path,
        metrics_log=tmp_path / "metrics.jsonl",
        hf_org="TestOrg",
        hf_upload=False,
    )
    orchestrator.uploader = mock_uploader

    def train_fn(key, ckpt_dir):
        return {"metrics": {"accuracy": 0.95}}

    results = orchestrator.run_zoo(
        keys=["test_mechanism"],
        train_fn=train_fn,
        key_field="mechanism_name",
    )
    assert len(results) == 1
    assert results[0]["status"] == "PASSED"
    assert mock_uploader.submit_upload.called
