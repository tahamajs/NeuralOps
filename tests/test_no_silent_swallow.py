#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests ensuring no silent swallowing of errors in data-producing paths."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_no_bare_except_pass_in_fixed_files():
    """Assert no except Exception: pass or bare except in P0 fixed sites/functions."""
    # 1. neuralops/api.py: replace_attention function
    api_path = ROOT / "neuralops" / "api.py"
    api_ast = ast.parse(api_path.read_text(encoding="utf-8"))
    for node in ast.walk(api_ast):
        if isinstance(node, ast.FunctionDef) and node.name == "replace_attention":
            for child in ast.walk(node):
                if isinstance(child, ast.ExceptHandler):
                    # Check for bare except:
                    assert child.type is not None, "Bare 'except:' found in replace_attention"
                    # Check for except Exception: pass
                    if len(child.body) == 1 and isinstance(child.body[0], ast.Pass):
                        pytest.fail(f"Silent 'except Exception: pass' found in replace_attention at line {child.lineno}")

    # 2. neuralops/main.py: run_eval and run_compress functions
    main_path = ROOT / "neuralops" / "main.py"
    main_ast = ast.parse(main_path.read_text(encoding="utf-8"))
    for node in ast.walk(main_ast):
        if isinstance(node, ast.FunctionDef) and node.name in ("run_eval", "run_compress"):
            for child in ast.walk(node):
                if isinstance(child, ast.ExceptHandler):
                    assert child.type is not None, f"Bare 'except:' found in {node.name}"
                    if len(child.body) == 1 and isinstance(child.body[0], ast.Pass):
                        pytest.fail(f"Silent 'except Exception: pass' found in {node.name} at line {child.lineno}")

    # 3. neuralops/export/exporter.py: load_exported_model function
    exp_path = ROOT / "neuralops" / "export" / "exporter.py"
    exp_ast = ast.parse(exp_path.read_text(encoding="utf-8"))
    for node in ast.walk(exp_ast):
        if isinstance(node, ast.FunctionDef) and node.name == "load_exported_model":
            for child in ast.walk(node):
                if isinstance(child, ast.ExceptHandler):
                    assert child.type is not None, "Bare 'except:' found in load_exported_model"
                    if len(child.body) == 1 and isinstance(child.body[0], ast.Pass):
                        pytest.fail(f"Silent 'except Exception: pass' found in load_exported_model at line {child.lineno}")


def test_run_eval_fails_on_checkpoint_failure():
    """Mock checkpoint loading to fail and assert run_eval raises without evaluating base model."""
    from neuralops.main import run_eval

    cfg = MagicMock()
    cfg.seed = 42
    cfg.model_name = "bert-base-uncased"
    cfg.batch_size = 8
    cfg.task = "sst2"

    logger = MagicMock()

    base_model_eval_mock = MagicMock()

    def mock_from_pretrained(path_or_name, *args, **kwargs):
        if path_or_name == "bad_checkpoint_path":
            raise OSError("Corrupt checkpoint files")
        # If it falls back to base model:
        base_model_eval_mock(path_or_name)
        return MagicMock()

    mock_auto_model = MagicMock()
    mock_auto_model.from_pretrained.side_effect = mock_from_pretrained

    mock_trainer = MagicMock()
    mock_trainer_cls = MagicMock(return_value=mock_trainer)

    fw = {
        "set_seed": MagicMock(),
        "get_device": MagicMock(return_value="cpu"),
        "AutoTokenizer": MagicMock(),
        "AutoModelForSequenceClassification": mock_auto_model,
        "TrainingArguments": MagicMock(),
        "Trainer": mock_trainer_cls,
        "DataCollatorWithPadding": MagicMock(),
        "get_task_metrics": MagicMock(),
        "ResultWriter": MagicMock(),
    }

    with patch("neuralops.main.load_data", return_value=({"validation": []}, 2, False)):
        with pytest.raises(RuntimeError) as exc_info:
            run_eval(cfg, fw, logger, model_path="bad_checkpoint_path")

    # Assert checkpoint path is in the failure message
    assert "bad_checkpoint_path" in str(exc_info.value)
    # Assert base model was never evaluated as a fallback
    assert base_model_eval_mock.call_count == 0
    assert mock_trainer.evaluate.call_count == 0


def test_load_exported_model_fails_on_weight_failure(tmp_path):
    """Mock safetensors load to fail and assert load_exported_model raises instead of continuing."""
    from neuralops.export.exporter import load_exported_model
    import json
    import torch.nn as nn

    export_dir = tmp_path / "exported"
    export_dir.mkdir()

    cfg_file = export_dir / "neuralops_config.json"
    cfg_file.write_text(json.dumps({
        "base_model": "test-base",
        "_neuralops_metadata": []
    }), encoding="utf-8")

    safetensors_path = export_dir / "model.safetensors"
    safetensors_path.write_bytes(b"corrupt")

    dummy_model = nn.Linear(10, 10)

    with patch("safetensors.torch.load_file", side_effect=RuntimeError("Header truncated")):
        with pytest.raises(RuntimeError) as exc_info:
            load_exported_model(export_dir, base_model=dummy_model)

    assert "model.safetensors" in str(exc_info.value) or "Header truncated" in str(exc_info.value)
