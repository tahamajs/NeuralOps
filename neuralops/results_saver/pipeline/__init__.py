# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Results Saver Pipeline Subpackage.
"""

from .result_writer import ResultWriter

__all__ = ["ResultWriter"]

from neuralops.results_saver.pipeline import result_writer
