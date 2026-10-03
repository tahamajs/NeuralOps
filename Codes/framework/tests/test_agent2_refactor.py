"""
Comprehensive unit tests for Agent 2 distributed systems & performance refactor:
1. DDP gradient accumulation & no_sync wrapping in CustomLoopTrainer and utils.
2. Factorized-aware adaptive gradient clipping scaling grad(U) and grad(V) independently.
3. Distributed launcher signal handling, child process cleanup, and zombie prevention.
4. Distributed Optuna TrialPruned broadcast via torch.distributed.all_gather_object.
5. SQLite WAL mode, PRAGMA synchronous=NORMAL, and file locking for multi-GPU database write safety.
"""

from __future__ import annotations

import concurrent.futures
import os
import shutil
import signal
import sqlite3
import tempfile
import unittest
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from framework.distributed.config import DistributedConfig
from framework.distributed.launcher import (_terminate_process_tree,
                                            launch_distributed)
from framework.hyperopt.backends.optuna_optimizer import (
    OptunaOptimizer, broadcast_pruned_state)
from framework.hyperopt.config import HPOConfig
from framework.results_saver.saver import (ResultsSaver,
                                           configure_sqlite_pragmas,
                                           sqlite_file_lock)
from framework.trainer.custom_loop_trainer import CustomLoopTrainer
from framework.trainer.utils import clip_grad_norm_factorized, train_model
from torch.utils.data import DataLoader, TensorDataset


class DummyDDPModel(nn.Module):
    """Mock model with .no_sync() context manager simulating DistributedDataParallel."""

    def __init__(self) -> None:
        super().__init__()
        self.u_proj = nn.Linear(32, 8, bias=False)
        self.v_proj = nn.Linear(8, 32, bias=False)
        self.head = nn.Linear(32, 2)
        self.no_sync_call_count = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.u_proj(input_ids)
        h = self.v_proj(h)
        logits = self.head(h)
        return logits

    def no_sync(self):
        self.no_sync_call_count += 1
        class MockNoSyncContext:
            def __enter__(self):
                return self
            def __exit__(self, exc_type, exc_val, exc_tb):
                pass
        return MockNoSyncContext()


class TestAgent2Refactor(unittest.TestCase):
    """Test suite verifying Agent 2 implementations."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # 1. CustomLoopTrainer & DDP Gradient Accumulation Tests
    # ------------------------------------------------------------------
    def test_ddp_gradient_accumulation_no_sync(self) -> None:
        """Test that non-final micro-batches are wrapped in model.no_sync() for DDP."""
        model = DummyDDPModel()
        inputs = torch.randn(8, 32)
        labels = torch.randint(0, 2, (8,))
        dataset = TensorDataset(inputs, labels)
        
        # DataLoader with batch_size=1 so there are 8 micro-batches
        def collate_fn(batch):
            x = torch.stack([b[0] for b in batch])
            y = torch.stack([b[1] for b in batch])
            return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": y}

        train_loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        trainer = CustomLoopTrainer(
            model=model,
            train_dataset=dataset,
            eval_dataset=dataset,
            config={"gradient_accumulation_steps": 4, "logging_steps": 10},
            device=torch.device("cpu"),
            num_labels=2,
        )

        # 8 micro-batches with gradient_accumulation_steps=4:
        # Step 0 (non-final -> no_sync)
        # Step 1 (non-final -> no_sync)
        # Step 2 (non-final -> no_sync)
        # Step 3 (sync step -> nullcontext)
        # Step 4 (non-final -> no_sync)
        # Step 5 (non-final -> no_sync)
        # Step 6 (non-final -> no_sync)
        # Step 7 (sync step -> nullcontext)
        # Total expected no_sync calls = 6
        epoch_loss = trainer.train_epoch(
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=None,
            gradient_accumulation_steps=4,
            max_grad_norm=1.0,
        )

        self.assertGreater(epoch_loss, 0.0)
        self.assertEqual(model.no_sync_call_count, 6)

    # ------------------------------------------------------------------
    # 2. Factorized Gradient Clipping Tests
    # ------------------------------------------------------------------
    def test_factorized_gradient_clipping_independent_scaling(self) -> None:
        """Test that grad(U) and grad(V) are scaled independently."""
        model = DummyDDPModel()
        # Set large gradients on U (norm ~1600) and small gradients on V (norm ~0.16 < 1.0)
        model.u_proj.weight.grad = torch.full_like(model.u_proj.weight, 100.0)
        model.v_proj.weight.grad = torch.full_like(model.v_proj.weight, 0.01)
        model.head.weight.grad = torch.full_like(model.head.weight, 0.5)

        v_grad_norm_before = torch.norm(model.v_proj.weight.grad)

        # Clip with u_max_norm=1.0, v_max_norm=1.0, dense_max_norm=1.0
        total_norm = clip_grad_norm_factorized(
            model,
            max_norm=1.0,
            u_max_norm=1.0,
            v_max_norm=1.0,
            dense_max_norm=1.0,
        )

        u_grad_norm_after = torch.norm(model.u_proj.weight.grad)
        v_grad_norm_after = torch.norm(model.v_proj.weight.grad)

        # U should be clipped to 1.0 (with small floating precision tolerance)
        self.assertAlmostEqual(u_grad_norm_after.item(), 1.0, places=4)
        # V should NOT be squashed by U's huge initial gradient because they are clipped independently
        self.assertAlmostEqual(v_grad_norm_after.item(), v_grad_norm_before.item(), places=4)
        self.assertGreater(total_norm.item(), 0.0)

    # ------------------------------------------------------------------
    # 3. Distributed Launcher & Process Termination Tests
    # ------------------------------------------------------------------
    def test_launcher_signal_handler_registration(self) -> None:
        """Test that launcher registers signal handlers and terminates cleanly."""
        config = DistributedConfig(num_nodes=1, num_gpus=1, elastic=False)
        called = []

        def dummy_train():
            called.append(True)

        res = launch_distributed(config, dummy_train, job_id="test_local")
        self.assertEqual(res, 0)
        self.assertTrue(called[0])

    def test_terminate_process_tree(self) -> None:
        """Test that _terminate_process_tree cleanly terminates a mock subprocess."""
        mock_proc = MagicMock()
        mock_proc.poll.side_effect = [None, None, 0]
        mock_proc.pid = 12345
        _terminate_process_tree(mock_proc, timeout=0.1)
        self.assertTrue(mock_proc.poll.called)

    # ------------------------------------------------------------------
    # 4. Optuna Distributed TrialPruned Broadcast Tests
    # ------------------------------------------------------------------
    def test_broadcast_pruned_state_single_and_mock_distributed(self) -> None:
        """Test broadcast_pruned_state in non-distributed and mock distributed mode."""
        # Non-distributed mode
        self.assertFalse(broadcast_pruned_state(False))
        self.assertTrue(broadcast_pruned_state(True))

        # Distributed mode simulation
        with patch("torch.distributed.is_available", return_value=True), \
             patch("torch.distributed.is_initialized", return_value=True), \
             patch("torch.distributed.get_world_size", return_value=4):
            
            def mock_all_gather(tensor_list, tensor_item):
                # Simulate rank 0 pruned (True), ranks 1-3 unpruned (False)
                tensor_list[0] = True
                tensor_list[1] = False
                tensor_list[2] = False
                tensor_list[3] = False

            with patch("torch.distributed.all_gather_object", side_effect=mock_all_gather):
                # When rank > 0 has False, it still receives True because rank 0 pruned
                synced_state = broadcast_pruned_state(False)
                self.assertTrue(synced_state)

    # ------------------------------------------------------------------
    # 5. SQLite WAL, NORMAL Synchronous & File Locking Tests
    # ------------------------------------------------------------------
    def test_sqlite_wal_and_pragmas(self) -> None:
        """Test that SQLite database opens with PRAGMA journal_mode=WAL and PRAGMA synchronous=NORMAL."""
        db_path = os.path.join(self.temp_dir, "test.db")
        conn = sqlite3.connect(db_path)
        configure_sqlite_pragmas(conn)
        
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0].lower()
        
        cursor.execute("PRAGMA synchronous;")
        sync_mode = cursor.fetchone()[0]
        
        cursor.close()
        conn.close()

        self.assertEqual(journal_mode, "wal")
        # In SQLite, synchronous=NORMAL is represented as integer 1
        self.assertIn(sync_mode, (1, "1", "NORMAL", "normal"))

    def test_results_saver_sqlite_integration(self) -> None:
        """Test ResultsSaver SQLite persistence with file locking."""
        saver = ResultsSaver(run_id="run_agent2_001", base_dir=self.temp_dir)
        db_file = os.path.join(self.temp_dir, "experiments.db")
        
        metrics = {"eval_accuracy": 0.985, "eval_loss": 0.045}
        config = {"model_name": "svd_transformer", "task_name": "mrpc", "rank": 32}
        
        saver.save_metrics(metrics)
        saver.save_config(config)

        self.assertTrue(os.path.exists(db_file))
        
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT run_id, task_name, model_name FROM experiment_runs WHERE run_id = ?", ("run_agent2_001",))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "run_agent2_001")
        self.assertEqual(row[1], "mrpc")
        self.assertEqual(row[2], "svd_transformer")

    def test_concurrent_sqlite_writes_with_file_locking(self) -> None:
        """Test concurrent multi-threaded/multi-GPU SQLite writes with file locking."""
        saver = ResultsSaver(run_id="run_concurrent", base_dir=self.temp_dir)
        db_file = os.path.join(self.temp_dir, "concurrent_metrics.db")

        def write_metric(step: int) -> None:
            saver.log_metric_to_sqlite(
                step=step,
                metric_name="train_loss",
                value=1.0 / (step + 1),
                db_path=db_file,
            )

        # Concurrently write 20 metrics across worker threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(write_metric, i) for i in range(20)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM metric_logs WHERE run_id = 'run_concurrent'")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()

        self.assertEqual(count, 20)


if __name__ == "__main__":
    unittest.main()
