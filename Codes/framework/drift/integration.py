"""Automatic integration with ResultsSaver and Django runner."""

from pathlib import Path
from typing import Dict, Any, Optional

from .detectors import DriftDetector
from .config import DriftConfig

def auto_drift_check(
    run_id: str,
    reference_data: Dict[str, Any],
    current_data: Dict[str, Any],
    config: Optional[DriftConfig] = None,
    output_dir: Path = None,
) -> "DriftReport":
    """
    Run drift detection and save the report to the run directory.
    Designed to be called from the Django runner's `final_callback` or
    from a dedicated monitoring job.

    Args:
        run_id: the training run ID
        reference_data: dict of feature → array for training data
        current_data: dict of feature → array for evaluation/serving data
        config: drift configuration (uses defaults if None)
        output_dir: base directory for results (e.g. settings.MEDIA_ROOT / run_id)

    Returns:
        DriftReport with all results
    """
    cfg = config or DriftConfig()
    detector = DriftDetector(reference_data, cfg)
    report = detector.detect(current_data)

    if output_dir:
        report.save(str(output_dir / "drift_report.json"))
        if cfg.generate_plots:
            from .visualization import generate_drift_plots
            generate_drift_plots(report, reference_data, current_data,
                                str(output_dir / "drift_plots"))

    return report