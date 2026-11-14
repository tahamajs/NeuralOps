#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Unit tests for Phase 10 Static Build generator."""

import json
from pathlib import Path

from neuralops.ui.static_builder import build_static_site


def test_build_static_site(tmp_path: Path):
    """Verify build_static_site exports complete pre-rendered HTML bundle with static assets."""
    out_dir = tmp_path / "static_dist"
    summary = build_static_site(out_dir)

    assert summary["status"] == "success"
    assert summary["total_pages"] >= 200
    assert out_dir.is_dir()

    # Core pages
    assert (out_dir / "index.html").is_file()
    assert (out_dir / "attentions.html").is_file()
    assert (out_dir / "compressors.html").is_file()
    assert (out_dir / "benchmarks.html").is_file()
    assert (out_dir / "models.html").is_file()
    assert (out_dir / "notebooks.html").is_file()
    assert (out_dir / "doctor.html").is_file()
    assert (out_dir / "manifest.json").is_file()

    # Static assets copied
    static_dir = out_dir / "static"
    assert static_dir.is_dir()
    assert (static_dir / "ui.css").is_file()
    assert (static_dir / "htmx.min.js").is_file()
    assert (static_dir / "plot.min.js").is_file()
    assert (static_dir / "charts.js").is_file()

    # Detail subdirectories
    assert (out_dir / "attentions" / "cosformer.html").is_file()
    assert (out_dir / "compressors" / "direct_svd.html").is_file()

    # Validate manifest
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["generator"] == "neuralops.ui.static_builder"
    assert manifest["total_pages"] == summary["total_pages"]
    assert "index.html" in manifest["files"]
    assert "attentions/cosformer.html" in manifest["files"]
