#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Griffin hybrid attention layer facade."""

from __future__ import annotations

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.layers.ssm.griffin_ssd import GriffinSSDBlock

GriffinBlock = GriffinSSDBlock
ATTENTION_REGISTRY.register("griffin", is_alias=True)(GriffinBlock)

__all__ = ["GriffinSSDBlock", "GriffinBlock"]
