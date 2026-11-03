#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit tests for the Universal Benchmark Pipeline, Adapter, Registry,
Mock Micro-Training Step, WAL Logging, and Resumption Logic.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict
import unittest

import torch
import torch.nn as nn

from neuralops.benchmarks.adapters.model import (
    BaseModelAdapter,
    UniversalModelAdapter,
)
from neuralops.benchmarks.eval_with_training import TrainAndBenchmarkPipeline
from neuralops.benchmarks.pipeline import BenchmarkPipeline
from neuralops.benchmarks.registry import (
    BENCHMARK_REGISTRY,
    BaseBenchmark,
    register_benchmark,
)
from neuralops.benchmarks.results.resumer import BenchmarkSweepResumer
from neuralops.benchmarks.results.saver import MultiFormatSaver
from neuralops.benchmarks.results.schema import BenchmarkResult
from neuralops.cli.bench_cmd import build_parser, handle_eval


class DummyBenchmarkModel(nn.Module):
    """Lightweight deterministic model for CPU testing."""

    def __init__(self, vocab_size: int = 100, embed_dim: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor, **kwargs: Any) -> Any:
        x = self.embed(input_ids)
        logits = self.fc(x)

        class Output:
            def __init__(self, l: torch.Tensor) -> None:
                self.logits = l

        return Output(logits)


@register_benchmark(name="dummy_mock_eval", mode="offline", hardware="cpu")
class DummyMockBenchmark(BaseBenchmark):
    """Registered mock benchmark runner."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(name="dummy_mock_eval", **kwargs)

    def run(self, model_adapter: BaseModelAdapter, **kwargs: Any) -> Dict[str, Any]:
        return {
            "dummy_metric": 42.0,
            "latency_ms": 1.25,
            "status": "PASS",
        }


class TestUniversalPipeline(unittest.TestCase):
    """Test suite verifying universal benchmark pipeline components."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp_dir.name)
        self.model = DummyBenchmarkModel()

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_universal_model_adapter_wrapping(self) -> None:
        """Verify UniversalModelAdapter properly wraps an arbitrary PyTorch nn.Module."""
        adapter = UniversalModelAdapter.from_pretrained(
            self.model,
            model_name="dummy_model",
            device="cpu",
        )
        self.assertIsInstance(adapter, BaseModelAdapter)
        self.assertEqual(adapter.model_name, "dummy_model")
        self.assertEqual(adapter.device.type, "cpu")

        # Test forward and get_logits
        input_ids = torch.randint(0, 100, (2, 8))
        logits = adapter.get_logits(input_ids)
        self.assertEqual(logits.shape, (2, 8, 100))

    def test_benchmark_registry_dispatch(self) -> None:
        """Verify BENCHMARK_REGISTRY decorator and lookup contract."""
        self.assertIn("dummy_mock_eval", BENCHMARK_REGISTRY)
        entry = BENCHMARK_REGISTRY["dummy_mock_eval"]
        self.assertEqual(entry["name"], "dummy_mock_eval")

        adapter = UniversalModelAdapter.from_pretrained(self.model, model_name="dummy_model")
        pipeline = BenchmarkPipeline(model_adapter=adapter, task_keys=["dummy_mock_eval"])
        res = pipeline.execute_task("dummy_mock_eval")
        self.assertIn("metrics", res)
        self.assertEqual(res["metrics"]["dummy_metric"], 42.0)

    def test_multiformatsaver_atomic_wal_and_ledger(self) -> None:
        """Verify MultiFormatSaver writes to raw_results, .journal.jsonl, and master_ledger.db."""
        saver = MultiFormatSaver(output_dir=self.output_dir)

        result = BenchmarkResult(
            run_id="test_run_001",
            model_name="dummy_model",
            benchmark_suite="dummy_mock_eval",
            device_type="cpu",
            status="success",
            metrics={"latency_ms": 1.5, "acc": 99.0},
            metadata={"regime": "mock"},
        )

        paths = saver.save(result)
        self.assertTrue(paths["raw_json"].exists())
        self.assertTrue(paths["journal"].exists())
        self.assertTrue(paths["summary_csv"].exists())
        self.assertTrue(paths["ledger_db"].exists())

        # Verify .journal.jsonl content
        with open(paths["journal"], "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["run_id"], "test_run_001")
        self.assertEqual(lines[0]["event"], "CASE_SUCCESS")

    def test_resumer_idempotency_audit(self) -> None:
        """Verify BenchmarkSweepResumer accurately detects completed cases from WAL journal."""
        saver = MultiFormatSaver(output_dir=self.output_dir)

        res1 = BenchmarkResult(
            run_id="run_a",
            model_name="model_x",
            transplanted_attention="fms_scan",
            benchmark_suite="dummy_mock_eval",
            status="success",
            metrics={"acc": 95.0},
        )
        saver.save(res1)

        resumer = BenchmarkSweepResumer(output_dir=self.output_dir)
        self.assertEqual(resumer.completed_count, 1)

        # Test should_skip logic
        skip, reason = resumer.should_skip("model_x", "fms_scan", "dummy_mock_eval")
        self.assertTrue(skip)
        self.assertIn("Already completed", reason)

        skip_new, _ = resumer.should_skip("model_x", "gla_attention", "dummy_mock_eval")
        self.assertFalse(skip_new)

    def test_mock_micro_training_step(self) -> None:
        """Verify micro_train_healer runs mock micro-training steps and reports loss reduction."""
        teacher = DummyBenchmarkModel()
        student = DummyBenchmarkModel()

        pipeline = TrainAndBenchmarkPipeline(output_dir=str(self.output_dir), device="cpu")

        # Synthetic batch dataloader
        class MockData(torch.utils.data.Dataset):
            def __len__(self) -> int:
                return 5

            def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
                return {
                    "input_ids": torch.randint(0, 100, (2, 4)),
                    "labels": torch.randint(0, 100, (2, 4)),
                }

        loader = torch.utils.data.DataLoader(MockData(), batch_size=1)
        healing_stats = pipeline.micro_train_healer(
            teacher=teacher,
            student=student,
            train_loader=loader,
            steps=3,
            optimizer_type="adamw",
            regime="kd_standard",
        )

        self.assertIn("initial_loss", healing_stats)
        self.assertIn("final_healed_loss", healing_stats)
        self.assertEqual(healing_stats["pre_train_steps"], 3)

    def test_cli_bench_cmd_eval_parsing(self) -> None:
        """Verify CLI bench_cmd argument parser and handle_eval dispatch."""
        parser = build_parser()
        args = parser.parse_args([
            "eval",
            "--model", "dummy_model",
            "--benchmarks", "dummy_mock_eval",
            "--output-dir", str(self.output_dir),
            "--device", "cpu",
        ])

        self.assertEqual(args.subcommand, "eval")
        self.assertEqual(args.model, "dummy_model")
        self.assertEqual(args.benchmarks, "dummy_mock_eval")

        # Execute handle_eval end-to-end
        ret = handle_eval(args)
        self.assertEqual(ret, 0)

        # Verify output directory has .journal.jsonl and master_ledger.db
        self.assertTrue((self.output_dir / ".journal.jsonl").exists())
        self.assertTrue((self.output_dir / "master_ledger.db").exists())


if __name__ == "__main__":
    unittest.main()
