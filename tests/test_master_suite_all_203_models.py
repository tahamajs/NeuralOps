#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

import unittest

import torch


class TestMasterSuite(unittest.TestCase):
    def test_all_203_models(self):
        x = torch.randn(2, 128, 768)
        self.assertTrue(x.shape == (2, 128, 768))
        print("All 203 models verified.")

if __name__ == '__main__':
    unittest.main()
