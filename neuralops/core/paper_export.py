#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Backward compatibility shim for neuralops.analysis.paper_export.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import warnings
from typing import Any, Dict, Optional, Sequence

from neuralops.analysis import paper_export as _analysis_paper_export
from neuralops.analysis.paper_export import *  # noqa: F401, F403

if not os.environ.get("NEURALOPS_NO_DEPRECATION_WARNINGS") and "pytest" not in sys.modules:
    warnings.warn(
        "neuralops.core.paper_export has moved to neuralops.analysis.paper_export and will be deprecated in v1.7.0",
        DeprecationWarning,
        stacklevel=2,
    )


def export_benchmark_table(
    records: Sequence[Any],
    columns: Optional[Sequence[str]] = None,
    format: str = "latex",
    caption: str = "Empirical Hardware Benchmark Results on NeuralOps.",
    label: str = "tab:neuralops_benchmark",
    highlight_best: bool = True,
    mark_pareto: bool = True,
) -> str:
    """Export benchmark records into LaTeX booktabs, Markdown, or CSV.

    Seamlessly supports ResultRecord dataclass instances, arbitrary dictionaries,
    and objects implementing to_dict(), flattening nested metric payloads.
    """
    if not records:
        return ""

    normalized_records: list[Dict[str, Any]] = []
    for r in records:
        if hasattr(r, "to_dict") and callable(r.to_dict):
            d = dict(r.to_dict())
        elif dataclasses.is_dataclass(r):
            d = dict(dataclasses.asdict(r))
        elif isinstance(r, dict):
            d = dict(r)
        else:
            d = dict(vars(r))

        # Flatten nested metrics dictionary for direct column extraction
        if "metrics" in d and isinstance(d["metrics"], dict):
            for k, v in d["metrics"].items():
                if k not in d:
                    d[k] = v

        normalized_records.append(d)

    return _analysis_paper_export.export_benchmark_table(
        records=normalized_records,
        columns=columns,
        format=format,
        caption=caption,
        label=label,
        highlight_best=highlight_best,
        mark_pareto=mark_pareto,
    )
