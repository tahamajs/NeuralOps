#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

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
