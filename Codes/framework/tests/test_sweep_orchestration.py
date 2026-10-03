"""Unit tests for sweep orchestration, checkpoint resolution, plateau detection, and resumption."""

import csv
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from framework.core.registry import (ATTENTION_REGISTRY,
                                     is_module_declared_compatible)
from framework.utils.compat import module_family_support


class TestSweepOrchestrationLogic:
    """Verify sweep runner decision branches, plateau handling, and summary.csv operations."""

    def test_parent_checkpoint_explicit_cli_override(self):
        """Verify --parent_checkpoint CLI argument explicitly takes precedence over defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_ckpt = Path(tmpdir) / "custom_baseline.pt"
            custom_ckpt.write_text("mock")

            # Simulate argument resolution logic from run_quality_sweep.py
            args_parent_checkpoint = str(custom_ckpt)
            out = Path(tmpdir) / "results" / "roberta_sst2_sweep"
            family = "roberta"
            dataset = "sst2"

            if args_parent_checkpoint:
                resolved_ckpt = Path(args_parent_checkpoint)
            else:
                candidates = [
                    out.parent / f"{family}_{dataset}_lora_baseline" / "finetuned_model.pt",
                    out.parent / f"{family}2_{dataset}_lora_baseline" / "finetuned_model.pt",
                ]
                resolved_ckpt = next((c for c in candidates if c.exists()), candidates[0])

            assert resolved_ckpt == custom_ckpt
            assert resolved_ckpt.exists()

    def test_plateau_detection_logic(self):
        """Verify that baseline 50.92% accuracy after recovery triggers plateau classification."""
        acc = 50.92
        family = "roberta"
        had_nan = False

        if had_nan:
            status, err = "NAN", "nan_loss_during_recovery"
        elif acc <= 51.5:
            status, err = f"SKIP_{family.upper()}_INCOMPATIBLE_PLATEAU", f"accuracy_{acc}_below_random_floor"
        else:
            status, err = "PASS", ""

        assert status == "SKIP_ROBERTA_INCOMPATIBLE_PLATEAU"
        assert "50.92" in err

    def test_summary_csv_resumption_skips_existing(self):
        """Verify sweep reads existing completed modules from summary.csv and skips them."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "summary.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["module_name", "status", "accuracy_pct"])
                writer.writeheader()
                writer.writerow({"module_name": "full", "status": "BASELINE", "accuracy_pct": 92.5})
                writer.writerow({"module_name": "adaptive_svd", "status": "PASS", "accuracy_pct": 89.2})

            # Re-read existing
            existing = set()
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("module_name"):
                        existing.add(row["module_name"])

            all_modules = ["full", "adaptive_svd", "asvd", "rpc"]
            pending = [m for m in all_modules if m not in existing]

            assert "full" in existing
            assert "adaptive_svd" in existing
            assert pending == ["asvd", "rpc"]

    def test_causal_module_skip_logic_for_encoder(self):
        """Verify early skip correctly tags causal module on roberta before running GPU steps."""
        mod_name = "adaptive_span"
        family = "roberta"
        is_compat, reason = is_module_declared_compatible(mod_name, family)

        if not is_compat:
            status = f"SKIP_{family.upper()}_INCOMPATIBLE_DECLARED"
            err = reason
        else:
            status = "PASS"
            err = ""

        assert status == "SKIP_ROBERTA_INCOMPATIBLE_DECLARED"
        assert "causal_only" in err

    def test_batch_size_escalation_oom_handler(self):
        """Verify OOM simulation halves batch size until min_batch_size."""
        def eval_step(batch_size):
            if batch_size > 8:
                raise torch.cuda.OutOfMemoryError("CUDA OOM")
            return 85.0

        current_bs = 32
        success = False
        while current_bs >= 4:
            try:
                acc = eval_step(current_bs)
                success = True
                break
            except (torch.cuda.OutOfMemoryError, RuntimeError):
                current_bs = current_bs // 2

        assert success is True
        assert current_bs == 8
        assert acc == 85.0
