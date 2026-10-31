#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Evidence Markdown Report Generator.
"""

from __future__ import annotations

from typing import List

from neuralops.reporting.evidence import EvidenceRecord


def render_report(evidence: EvidenceRecord) -> str:
    """Format EvidenceRecord as human-readable GitHub-flavored Markdown report."""
    status_icon = "✓ PASS" if evidence.success else "✗ FAIL"
    lines: List[str] = [
        f"# NeuralOps Transformation Report: {evidence.recipe_name}",
        "",
        f"**Status:** {status_icon} | **Timestamp:** `{evidence.timestamp}` | **Recipe Hash:** `{evidence.recipe_hash[:16]}...`",
        "",
        "## Executive Summary",
        "",
        f"- **Target Model:** `{evidence.model_source}`",
        f"- **Stages Executed:** {evidence.metrics.get('stages_executed', 0)}",
        f"- **Total Duration:** {evidence.metrics.get('duration_seconds', 0.0):.2f}s",
    ]

    comp_ratio = evidence.metrics.get("compression_ratio")
    if comp_ratio is not None:
        lines.append(f"- **Final Compression Ratio:** {comp_ratio:.4f}")

    lines.extend([
        "",
        "## Environment",
        "",
        f"- **Python:** `{evidence.environment.get('python', {}).get('version')}`",
        f"- **PyTorch:** `{evidence.environment.get('torch', {}).get('version')}`",
        f"- **Platform:** `{evidence.environment.get('platform', {}).get('system')} {evidence.environment.get('platform', {}).get('machine')}`",
        f"- **Accelerator:** `{evidence.environment.get('torch', {}).get('device_name')}`",
        f"- **NeuralOps:** `v{evidence.environment.get('neuralops', {}).get('version')}` (`{evidence.environment.get('neuralops', {}).get('git_commit')[:8]}`)",
        "",
        "## Stage Details",
        "",
        "| Stage ID | Operation | Status | Duration | Metrics |",
        "| :--- | :--- | :---: | :---: | :--- |",
    ])

    for s in evidence.stages:
        st = "✓ OK" if s.get("success") else "✗ FAIL"
        ev = s.get("evidence", {})
        det = []
        if "compression_ratio" in ev:
            det.append(f"ratio={ev['compression_ratio']:.4f}")
        if "modules_compressed" in ev:
            det.append(f"mods={ev['modules_compressed']}")
        metrics_str = ", ".join(det) if det else "-"
        lines.append(f"| `{s.get('stage_id')}` | `{s.get('operation_type')}` | {st} | {s.get('duration_seconds', 0.0):.2f}s | {metrics_str} |")

    if evidence.validation:
        val = evidence.validation
        lines.extend([
            "",
            "## Validation Gates",
            "",
            f"**Profile:** `{val.get('profile')}` | **Gates Passed:** {val.get('passed_gates')}/{val.get('total_gates')} | **Duration:** {val.get('duration_seconds', 0.0):.3f}s",
            "",
            "| Gate Name | Severity | Passed | Message |",
            "| :--- | :---: | :---: | :--- |",
        ])
        for g in val.get("gates", []):
            passed = "✓" if g.get("passed") else "✗"
            lines.append(f"| `{g.get('gate_name')}` | `{g.get('severity')}` | {passed} | {g.get('message')} |")

    lines.append("")
    return "\n".join(lines)


__all__ = ["render_report"]
