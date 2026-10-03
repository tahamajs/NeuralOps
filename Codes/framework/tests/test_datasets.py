"""
Unit tests for benchmark dataset registry, loaders, and synthetic generation.
"""

import unittest

import torch
from framework.data.benchmark_datasets import (BenchmarkDatasetRegistry,
                                               get_synthetic_benchmark_sample,
                                               load_benchmark_dataset)


class TestBenchmarkDatasets(unittest.TestCase):
    """Test suite verifying dataset loading, batch collation, and synthetic fallback."""

    def test_registry_populated(self):
        suites = BenchmarkDatasetRegistry.list_benchmarks()
        self.assertGreaterEqual(len(suites), 10)
        self.assertIn("wikitext2", suites)
        self.assertIn("c4", suites)
        self.assertIn("gsm8k", suites)
        self.assertIn("humaneval", suites)
        self.assertIn("mmlu", suites)

    def test_synthetic_sample_generation(self):
        samples = get_synthetic_benchmark_sample("wikitext2", num_samples=5, seed=42)
        self.assertEqual(len(samples), 5)
        self.assertTrue(all(isinstance(s, str) and len(s) > 0 for s in samples))

    def test_load_benchmark_dataset_raw_samples(self):
        samples = load_benchmark_dataset(
            "wikitext2",
            split="validation",
            max_samples=4,
            use_synthetic=True,
            return_dataloader=False,
        )
        self.assertEqual(len(samples), 4)
        item = samples[0]
        self.assertIn("text", item)
        self.assertIsInstance(item["text"], str)

    def test_load_benchmark_dataset_dataloader(self):
        loader = load_benchmark_dataset(
            "wikitext2",
            batch_size=2,
            max_samples=4,
            use_synthetic=True,
            return_dataloader=True,
        )
        self.assertGreaterEqual(len(loader), 1)
        batch = next(iter(loader))
        self.assertIn("text", batch)
        self.assertEqual(len(batch["text"]), 2)

    def test_load_multiple_benchmark_splits(self):
        for name in ["ptb", "pile", "glue_sst2", "humaneval"]:
            ds = load_benchmark_dataset(
                name,
                max_samples=2,
                use_synthetic=True,
                return_dataloader=False,
            )
            self.assertEqual(len(ds), 2)


if __name__ == "__main__":
    unittest.main()
