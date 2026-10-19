#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Monitor HTML Rendering

Renders responsive HTML UI templates and visual dashboards for experiment tracking.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


def render_favicon() -> bytes:
    """Return raw SVG bytes for favicon icon."""
    return b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">\xe2\x9a\xa1</text></svg>'


def render_csv_export(results: list[dict[str, Any]]) -> bytes:
    """Export benchmark records to CSV format bytes."""
    df = pd.DataFrame(results)
    csv_str = df.to_csv(index=False)
    return csv_str.encode("utf-8")


def render_json_export(stats: dict[str, Any]) -> bytes:
    """Export benchmark statistics to JSON format bytes."""
    return json.dumps(stats, indent=2).encode("utf-8")


__all__ = [
    "render_favicon",
    "render_csv_export",
    "render_json_export",
]
