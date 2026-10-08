"""HTML rendering and response serialization utilities for monitor server."""

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
