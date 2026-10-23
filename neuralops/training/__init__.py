#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Backward compatibility facade for neuralops.training.
All training components have moved to neuralops.trainer.
"""

from __future__ import annotations

import sys
import warnings
from neuralops.trainer import *  # noqa: F401, F403
import neuralops.trainer.checkpointing as _chk  # noqa: F401
import neuralops.trainer.layer_distill as _ld  # noqa: F401
import neuralops.trainer.trainer as _tr  # noqa: F401

sys.modules["neuralops.training.checkpointing"] = _chk
sys.modules["neuralops.training.layer_distill"] = _ld
sys.modules["neuralops.training.trainer"] = _tr

warnings.warn(
    "neuralops.training has merged into neuralops.trainer and will be removed in v1.7.0.",
    DeprecationWarning,
    stacklevel=2,
)
