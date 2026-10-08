"""Tests for runner modules and monitor server functionality."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from neuralops.runners.run_monitor_server import (
    ComprehensiveStudioHandler,
    PORT,
    PROJECT_DIR,
    build_aggregated_stats,
    load_harvested_benchmark_records,
)
from neuralops.runners.unified_orchestrator import MODELS, PASS_THRESHOLD


class TestMonitorServer:
    """Test suite for the monitor server and stats aggregation."""

    def test_module_constants(self):
        assert PORT == 8080
        assert isinstance(PROJECT_DIR, str)

    def test_build_aggregated_stats(self, tmp_path):
        harvested_file = tmp_path / "harvested.json"
        harvested_file.write_text("[]")
        docs_dir = tmp_path / "Docs"
        docs_dir.mkdir()

        stats = build_aggregated_stats(
            project_dir=str(tmp_path),
            harvested_json_path=str(harvested_file),
            docs_dir=str(docs_dir),
            is_local_gpu=False,
        )
        assert isinstance(stats, dict)
        assert "counts" in stats
        assert "gpu" in stats

    def test_load_harvested_benchmark_records_missing(self, tmp_path):
        records = load_harvested_benchmark_records(
            project_dir=str(tmp_path),
            harvested_json_path=str(tmp_path / "nonexistent.json"),
        )
        assert isinstance(records, list)

    def test_studio_handler_attributes(self):
        assert hasattr(ComprehensiveStudioHandler, "project_dir")
        assert hasattr(ComprehensiveStudioHandler, "web_dir")
        assert hasattr(ComprehensiveStudioHandler, "docs_dir")


class TestUnifiedOrchestrator:
    """Test suite for the unified orchestrator runner."""

    def test_models_config(self):
        assert isinstance(MODELS, list)
        assert len(MODELS) >= 4
        for m in MODELS:
            assert "name" in m
            assert "slug" in m
            assert "family" in m

    def test_pass_threshold(self):
        assert isinstance(PASS_THRESHOLD, (int, float))
        assert PASS_THRESHOLD > 0
