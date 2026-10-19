#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import unittest
import importlib

class TestEvaluationImports(unittest.TestCase):
    def test_imports(self):
        modules = [
            "evaluation.ab_testing",
            "evaluation.eval_all_models_datasets",
            "evaluation.eval_data",
            "evaluation.eval_patcher",
            "evaluation.eval_perplexity",
            "evaluation.eval_pipeline",
            "evaluation.eval_reporter",
            "evaluation.eval_types",
            "evaluation.eval_utils",
            "evaluation.fairness",
            "evaluation.full_mnli_validation",
            "evaluation.label_quality",
            "evaluation.metrics",
            "evaluation.retraining",
            "evaluation.run_baseline_comparison",
            "evaluation.run_lm_eval_harness",
            "evaluation.harness",
            "evaluation.ppl",
            "evaluation.niah",
            "evaluation.niah_passkey",
            "evaluation.lm_eval_adapter",
        ]
        for mod in modules:
            with self.subTest(module=mod):
                m = importlib.import_module(mod)
                self.assertIsNotNone(m)

    def test_backwards_compatible_shims(self):
        legacy_modules = [
            "neuralops.evaluation.ab_testing",
            "neuralops.evaluation.eval_all_models_datasets",
            "neuralops.evaluation.eval_data",
            "neuralops.evaluation.eval_patcher",
            "neuralops.evaluation.eval_perplexity",
            "neuralops.evaluation.eval_pipeline",
            "neuralops.evaluation.eval_reporter",
            "neuralops.evaluation.eval_types",
            "neuralops.evaluation.eval_utils",
            "neuralops.evaluation.fairness",
            "neuralops.evaluation.full_mnli_validation",
            "neuralops.evaluation.label_quality",
            "neuralops.evaluation.metrics",
            "neuralops.evaluation.retraining",
            "neuralops.evaluation.run_baseline_comparison",
            "neuralops.evaluation.run_lm_eval_harness",
        ]
        for mod in legacy_modules:
            with self.subTest(legacy_module=mod):
                m = importlib.import_module(mod)
                self.assertIsNotNone(m)

if __name__ == "__main__":
    unittest.main()
