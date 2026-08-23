"""Auto‑drift check integration – called by MegaOrchestrator post‑training."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .config import DriftConfig
from .detectors import DriftDetector, DriftReport


def auto_drift_check(
    run_id: str,
    reference_data: Dict[str, np.ndarray],
    current_data: Dict[str, np.ndarray],
    config: Optional[DriftConfig] = None,
    output_dir: Optional[Path] = None,
) -> "DriftReport":
    """Auto drift check."""
    cfg = config or DriftConfig()
    detector = DriftDetector(reference_data, cfg)
    report = detector.detect(current_data)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.save(str(output_dir / "drift_report.json"))
        if cfg.generate_plots:
            from .visualization import generate_drift_plots

            generate_drift_plots(
                report, reference_data, current_data, str(output_dir / "drift_plots")
            )

    return report
