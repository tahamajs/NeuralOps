import unittest
import importlib

class TestEvaluationImports(unittest.TestCase):
    def test_imports(self):
        modules = [
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
        for mod in modules:
            with self.subTest(module=mod):
                m = importlib.import_module(mod)
                self.assertIsNotNone(m)

if __name__ == "__main__":
    unittest.main()
