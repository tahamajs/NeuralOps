import unittest
import importlib

class TestRunnersImports(unittest.TestCase):
    def test_imports(self):
        modules = [
            "neuralops.runners.run_master_pipeline",
            "neuralops.runners.run_sweep",
            "neuralops.runners.unified_orchestrator",
        ]
        for mod in modules:
            with self.subTest(module=mod):
                m = importlib.import_module(mod)
                self.assertIsNotNone(m)

if __name__ == "__main__":
    unittest.main()
