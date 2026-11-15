"""Verification test suite for Wave 8 W8-3 Naked Print cleanout.

Verifies that zero naked print statements remain in core subsystems:
- evaluation/standalone
- runners
- hub
- trainer

Also verifies that logging functions correctly in these subsystems.
"""

import ast
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_zero_naked_prints_in_target_subsystems():
    """Verify via AST parsing that no Call nodes invoke print() in target subsystems."""
    root_dir = Path(__file__).resolve().parent.parent.parent / "neuralops"
    target_subdirs = [
        root_dir / "evaluation" / "standalone",
        root_dir / "runners",
        root_dir / "hub",
        root_dir / "trainer",
    ]

    naked_prints = []

    for subdir in target_subdirs:
        assert subdir.exists(), f"Target directory {subdir} does not exist"
        for py_file in subdir.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    if node.func.id == "print":
                        naked_prints.append((str(py_file.relative_to(root_dir.parent)), node.lineno))

    assert not naked_prints, f"Found naked print() calls: {naked_prints}"


def test_standalone_print_log_delegates_to_logger(caplog):
    """Verify that print_log in standalone evaluation scripts dispatches to logger.info."""
    from neuralops.evaluation.standalone.eval_ultra_long_context_scaling import print_log

    with caplog.at_level(logging.INFO):
        print_log("Test ultra long context message")

    assert "Test ultra long context message" in caplog.text


def test_monitor_server_uses_logger(caplog):
    """Verify that monitor server dispatches startup logs through logger."""
    import neuralops.runners.monitor.server as srv

    with patch("uvicorn.run") as mock_uvicorn:
        with patch("neuralops.runners.monitor.fastapi_app.create_app") as mock_app:
            mock_app.return_value = MagicMock()
            with caplog.at_level(logging.INFO):
                srv.run_server(port=9999, host="127.0.0.1", use_fastapi=True)
            mock_uvicorn.assert_called_once()

    assert "Starting FastAPI + WebSocket Telemetry Server on http://127.0.0.1:9999" in caplog.text


def test_monitor_stats_error_logged(caplog):
    """Verify that errors in harvested json loading log an error rather than printing."""
    from neuralops.runners.monitor.stats import load_harvested_benchmark_records

    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", side_effect=ValueError("Corrupt JSON")):
            with caplog.at_level(logging.ERROR):
                records = load_harvested_benchmark_records("fake_results", "fake_harvest.json")

    assert "Error loading fake_harvest.json" in caplog.text


def test_hyperopt_logging(caplog):
    """Verify HyperparameterOptimizer uses logger.info upon trial completion."""
    import neuralops.trainer.hyperopt as hopt
    from neuralops.trainer.hyperopt import HyperparameterOptimizer

    # Mock optuna study
    mock_study = MagicMock()
    mock_study.best_value = 0.9876
    mock_study.best_params = {"lr": 1e-4}
    mock_study.trials_dataframe.return_value = MagicMock()

    mock_optuna = MagicMock()
    mock_optuna.create_study.return_value = mock_study

    with patch.object(hopt, "_HAS_OPTUNA", True):
        with patch.object(hopt, "optuna", mock_optuna):
            optimizer = HyperparameterOptimizer(
                train_fn=lambda x: None,
                eval_fn=lambda x: None,
                config_template={},
            )
            optimizer.study = mock_study

            with patch("builtins.open", MagicMock()):
                with caplog.at_level(logging.INFO):
                    res = optimizer.run(n_trials=1)

    assert res == {"lr": 1e-4}
    assert "Optimization complete! Best value: 0.9876" in caplog.text
    assert "Best params: {'lr': 0.0001}" in caplog.text
