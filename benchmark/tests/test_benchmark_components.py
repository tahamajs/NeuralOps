"""
Comprehensive unit test suite for the Attention Benchmark Suite (ABS).
Validates metrics, masks, datasets, collators, diagnostics, model factories, profiling, and reporting.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import tempfile
import json
import os

from benchmark.core.metrics import compute_metrics
from benchmark.core.mask_utils import convert_mask_to_bool
from benchmark.core.seed import set_seed
from benchmark.core.model_factory import build_model_with_attention, get_transformer_layers, extract_attention_outputs
from benchmark.tasks.collator import DataCollatorForGLUE
from benchmark.tasks.dataset_loader import create_synthetic_glue_dataset
from benchmark.tasks.glue_tasks import get_task_info, GLUE_TASK_CONFIGS
from benchmark.diagnostics.attention_analysis import compute_attention_entropy, compute_attention_sparsity
from benchmark.diagnostics.gradient_analysis import compute_gradient_norm_stats
from benchmark.diagnostics.moe_analysis import compute_moe_load_balance
from benchmark.utils.flops import compute_flops
from benchmark.utils.memory import measure_gpu_memory, get_cpu_memory_mb
from benchmark.utils.timeouts import TimeoutException, timeout
from benchmark.report_generator import calculate_scores, generate_summary_table, generate_leaderboard_markdown


class TestMetrics(unittest.TestCase):
    def test_sst2_accuracy(self):
        preds = [1, 0, 1, 1]
        labels = [1, 0, 0, 1]
        res = compute_metrics("sst2", preds, labels)
        self.assertIn("accuracy", res)
        self.assertEqual(res["accuracy"], 0.75)
        self.assertEqual(res["score"], 0.75)

    def test_mrpc_f1(self):
        preds = [1, 0, 1, 1]
        labels = [1, 0, 1, 0]
        res = compute_metrics("mrpc", preds, labels)
        self.assertIn("f1", res)
        self.assertIn("accuracy", res)
        self.assertIn("score", res)

    def test_qqp_f1(self):
        preds = [1, 0, 1, 0]
        labels = [1, 0, 1, 1]
        res = compute_metrics("qqp", preds, labels)
        self.assertIn("f1", res)
        self.assertIn("accuracy", res)

    def test_cola_mcc(self):
        preds = [1, 0, 1, 0]
        labels = [1, 0, 1, 0]
        res = compute_metrics("cola", preds, labels)
        self.assertIn("mcc", res)
        self.assertAlmostEqual(res["mcc"], 1.0)
        self.assertAlmostEqual(res["score"], 1.0)

    def test_stsb_correlation(self):
        preds = [1.0, 2.0, 3.0, 4.0]
        labels = [1.1, 2.1, 2.9, 4.2]
        res = compute_metrics("stsb", preds, labels)
        self.assertIn("pearson", res)
        self.assertIn("spearman", res)
        self.assertIn("score", res)
        self.assertGreater(res["pearson"], 0.9)

    def test_multiclass_mnli(self):
        preds = [0, 1, 2, 0]
        labels = [0, 1, 2, 1]
        res = compute_metrics("mnli", preds, labels)
        self.assertIn("accuracy", res)
        self.assertEqual(res["accuracy"], 0.75)

    def test_empty_predictions_fallback(self):
        res = compute_metrics("sst2", [], [])
        self.assertEqual(res["accuracy"], 0.0)


class TestMaskUtils(unittest.TestCase):
    def test_2d_mask(self):
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        b_mask = convert_mask_to_bool(mask, B=2, L=3, device=torch.device("cpu"))
        self.assertEqual(b_mask.shape, (2, 3))
        self.assertTrue(b_mask[0, 0].item())
        self.assertFalse(b_mask[0, 2].item())

    def test_3d_mask(self):
        mask = torch.ones((2, 1, 3))
        mask[:, :, -1] = 0
        b_mask = convert_mask_to_bool(mask, B=2, L=3, device=torch.device("cpu"))
        self.assertEqual(b_mask.shape, (2, 3))
        self.assertTrue(b_mask[0, 0].item())
        self.assertFalse(b_mask[0, 2].item())

    def test_4d_additive_mask(self):
        mask = torch.zeros((2, 1, 1, 3))
        mask[:, :, :, -1] = -10000.0
        b_mask = convert_mask_to_bool(mask, B=2, L=3, device=torch.device("cpu"))
        self.assertEqual(b_mask.shape, (2, 3))
        self.assertTrue(b_mask[0, 0].item())

    def test_none_mask_fallback(self):
        b_mask = convert_mask_to_bool(None, B=2, L=4, device=torch.device("cpu"))
        self.assertEqual(b_mask.shape, (2, 4))
        self.assertTrue(b_mask.all().item())


class TestCollator(unittest.TestCase):
    class DummyTokenizer:
        def pad(self, inputs, padding=True, return_tensors="pt"):
            max_len = max(len(x) for x in inputs["input_ids"])
            padded_ids = [x + [0] * (max_len - len(x)) for x in inputs["input_ids"]]
            padded_mask = [x + [0] * (max_len - len(x)) for x in inputs["attention_mask"]]
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            }

    def test_classification_collator(self):
        tokenizer = self.DummyTokenizer()
        collator = DataCollatorForGLUE(tokenizer=tokenizer, is_regression=False)
        features = [
            {"input_ids": [101, 2054, 102], "labels": 1},
            {"input_ids": [101, 102], "labels": 0},
        ]
        batch = collator(features)
        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["labels"].dtype, torch.long)

    def test_regression_collator(self):
        tokenizer = self.DummyTokenizer()
        collator = DataCollatorForGLUE(tokenizer=tokenizer, is_regression=True)
        features = [
            {"input_ids": [101, 102], "label": 3.4},
        ]
        batch = collator(features)
        self.assertEqual(batch["labels"].dtype, torch.float32)


class TestDiagnostics(unittest.TestCase):
    def test_attention_entropy_and_sparsity(self):
        # Uniform attention weights
        weights = torch.ones((2, 4, 8, 8)) / 8.0
        entropy = compute_attention_entropy(weights)
        self.assertGreater(entropy, 0.0)

        # Sparse attention weights
        sparse_weights = torch.zeros((2, 4, 8, 8))
        sparse_weights[:, :, :, 0] = 1.0
        sparsity = compute_attention_sparsity(sparse_weights, threshold=1e-3)
        self.assertGreater(sparsity, 50.0)

    def test_none_attention_diagnostics(self):
        self.assertEqual(compute_attention_entropy(None), 0.0)
        self.assertEqual(compute_attention_sparsity(None), 0.0)

    def test_gradient_analysis(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 2))
        inputs = {"input": torch.randn(4, 16)}

        class Wrapper(nn.Module):
            def __init__(self, net):
                super().__init__()
                self.net = net

            def forward(self, input):
                return self.net(input)

        stats = compute_gradient_norm_stats(Wrapper(model), inputs)
        self.assertIn("mean_norm", stats)
        self.assertFalse(stats["has_nan"])

    def test_moe_load_balance_no_moe(self):
        model = nn.Linear(16, 16)
        inputs = {"input": torch.randn(4, 16)}
        stats = compute_moe_load_balance(model, inputs)
        self.assertFalse(stats["has_moe"])


class TestSyntheticDataAndTasks(unittest.TestCase):
    def test_synthetic_glue_dataset_all_tasks(self):
        for task in GLUE_TASK_CONFIGS.keys():
            train_ds, eval_ds, num_labels = create_synthetic_glue_dataset(task, num_samples=10)
            self.assertEqual(len(train_ds), 10)
            self.assertEqual(len(eval_ds), 5)
            self.assertGreaterEqual(num_labels, 1)

    def test_get_task_info(self):
        self.assertEqual(get_task_info("sst-2")["hf_name"], "sst2")
        self.assertEqual(get_task_info("STS-B")["is_regression"], True)
        self.assertEqual(get_task_info("unknown_task")["hf_name"], "sst2")


class TestUtilities(unittest.TestCase):
    def test_set_seed(self):
        set_seed(42)
        r1 = np.random.rand()
        set_seed(42)
        r2 = np.random.rand()
        self.assertEqual(r1, r2)

    def test_compute_flops(self):
        model = nn.Sequential(nn.Linear(32, 32))
        inputs = {"input_ids": torch.randint(0, 100, (2, 16))}
        flops = compute_flops(model, inputs)
        self.assertIsNotNone(flops)
        self.assertGreater(flops, 0.0)

    def test_memory_utils(self):
        cpu_mem = get_cpu_memory_mb()
        self.assertGreaterEqual(cpu_mem, 0.0)
        gpu_mem = measure_gpu_memory()
        self.assertGreaterEqual(gpu_mem, 0.0)


class TestReporting(unittest.TestCase):
    def test_calculate_scores_and_leaderboard(self):
        results = {
            "full": {
                "status": "OK",
                "avg_glue": 0.85,
                "num_params": 66000000,
                "profiling": {
                    "configurations": {
                        "bs32_seq512": {"tokens_per_sec": 12000.0, "peak_vram_mb": 500.0}
                    }
                },
                "stability": {
                    "total_seeds": 3,
                    "failed_seeds": 0,
                    "cv_lr": 0.05,
                    "stable": True,
                },
                "glue_tasks": {
                    "sst2": {"eval_metrics": {"accuracy": 0.90, "score": 0.90}},
                    "mrpc": {"eval_metrics": {"f1": 0.85, "score": 0.85}},
                }
            },
            "svd_linear": {
                "status": "OK",
                "avg_glue": 0.83,
                "num_params": 64000000,
                "profiling": {
                    "configurations": {
                        "bs32_seq512": {"tokens_per_sec": 18000.0, "peak_vram_mb": 350.0}
                    }
                },
                "stability": {
                    "total_seeds": 3,
                    "failed_seeds": 0,
                    "cv_lr": 0.08,
                    "stable": True,
                },
                "glue_tasks": {
                    "sst2": {"eval_metrics": {"accuracy": 0.88, "score": 0.88}},
                    "mrpc": {"eval_metrics": {"f1": 0.83, "score": 0.83}},
                }
            }
        }
        scores = calculate_scores(results)
        self.assertIn("full", scores)
        self.assertIn("svd_linear", scores)
        self.assertGreater(scores["full"]["total_score"], 0.0)
        self.assertGreater(scores["svd_linear"]["total_score"], 0.0)

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "summary.csv")
            md_path = os.path.join(tmp_dir, "leaderboard.md")
            df = generate_summary_table(results, scores, csv_path)
            generate_leaderboard_markdown(df, md_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(md_path))
            self.assertEqual(len(df), 2)


if __name__ == "__main__":
    unittest.main()
