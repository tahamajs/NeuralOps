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

from pathlib import Path
from typing import Any, Dict, List, Union


def render_report(evidence: Dict[str, Any]) -> str:
    """Format evidence dictionary as human-readable GitHub-flavored Markdown report.

    Contains required sections: Model, Environment, Stages, Repro.
    """
    status = evidence.get("status", "UNKNOWN")
    status_icon = "✓ PASS" if status == "SUCCESS" else ("⚠ PARTIAL" if status == "PARTIAL" else "✗ FAIL")
    recipe_hash = str(evidence.get("recipe_hash", "unknown"))
    short_hash = recipe_hash[:16] + "..." if len(recipe_hash) > 16 else recipe_hash
    recipe_name = evidence.get("recipe_name")
    header_title = (
        f"# NeuralOps Transformation Report: {recipe_name}"
        if recipe_name and recipe_name != "unnamed_recipe"
        else "# NeuralOps Transformation Evidence Report"
    )

    lines: List[str] = [
        header_title,
        "",
        f"**Status:** {status_icon} | **Recipe Hash:** `{short_hash}` | **Total Time:** {evidence.get('total_time_seconds', 0.0):.2f}s",
        "",
        "## Model",
        "",
    ]

    model_info = evidence.get("model", {})
    metrics = evidence.get("metrics", {})
    comp_ratio = metrics.get("compression_ratio")

    lines.extend([
        f"- **Source:** `{model_info.get('source', 'unknown')}`",
        f"- **Revision:** `{model_info.get('revision', 'main')}`",
        f"- **Final Parameter Count:** {model_info.get('final_parameter_count', 0):,}",
    ])
    if comp_ratio is not None:
        lines.append(f"- **Final Compression Ratio:** {comp_ratio:.4f}")

    lines.extend([
        "",
        "## Environment",
        "",
    ])

    env = evidence.get("environment", {})
    lines.extend([
        f"- **Hostname:** `{env.get('hostname', 'unknown')}`",
        f"- **NeuralOps:** `v{env.get('neuralops', 'unknown')}` (`{str(env.get('git_sha', 'unknown'))[:8]}`)",
        f"- **PyTorch:** `{env.get('torch', 'unknown')}`",
        f"- **CUDA:** `{env.get('cuda', 'None (CPU)')}`",
        f"- **GPU:** `{env.get('gpu', 'None')}`",
        f"- **Transformers:** `{env.get('transformers', 'Not installed')}`",
        "",
        "## Stages",
        "",
        "| Stage ID | Operation | Status | Duration | Gates Passed | Metrics |",
        "| :--- | :--- | :---: | :---: | :--- | :--- |",
    ])

    stages = evidence.get("stages", [])
    for s in stages:
        st_id = s.get("stage_id", "s")
        op_name = s.get("operation_name", s.get("operation_type", "unknown"))
        ok = "✓ OK" if s.get("success", True) else "✗ FAIL"
        dur = float(s.get("duration", s.get("duration_seconds", 0.0)))
        vr = s.get("validation_result", {})
        gates_str = "-"
        if isinstance(vr, dict) and "total_gates" in vr:
            gates_str = f"{vr.get('passed_gates', 0)}/{vr.get('total_gates', 0)}"

        ev = s.get("evidence", {})
        det = []
        if isinstance(ev, dict):
            if "compression_ratio" in ev:
                det.append(f"ratio={float(ev['compression_ratio']):.4f}")
            if "modules_compressed" in ev:
                det.append(f"mods={ev['modules_compressed']}")
        metrics_str = ", ".join(det) if det else "-"

        lines.append(f"| `{st_id}` | `{op_name}` | {ok} | {dur:.2f}s | {gates_str} | {metrics_str} |")

    lines.extend([
        "",
        "## Repro",
        "",
    ])

    repro = evidence.get("reproducibility", {})
    lines.extend([
        f"- **Language Version:** `{repro.get('language_version', 'v1.0.0')}`",
        f"- **Seed:** `{repro.get('seed', 'None')}`",
        f"- **Deterministic:** `{repro.get('deterministic', True)}`",
        f"- **Operations:** {repro.get('op_versions', {})}",
        "",
    ])

    return "\n".join(lines)


def save_report(
    evidence: Dict[str, Any],
    path: Union[str, Path],
    format: str = "markdown",
) -> None:
    """Save formatted evidence report to disk."""
    if format.lower() != "markdown":
        raise ValueError(f"Unsupported report format: {format}. Supported: 'markdown'")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    content = render_report(evidence)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)


__all__ = ["render_report", "save_report"]
