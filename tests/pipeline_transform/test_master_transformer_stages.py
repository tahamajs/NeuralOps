#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for MasterModelTransformer stages.
Ensures unimplemented stages raise NotImplementedError instead of being silent no-ops.
"""

import pytest
import torch.nn as nn

from neuralops.pipeline_transform.master_transformer import MasterTransformerPipeline


def test_unimplemented_stages_raise_not_implemented_error():
    """Verify execute_stage_svd, execute_stage_prune, and execute_stage_quantize raise NotImplementedError."""
    transformer = MasterTransformerPipeline()
    model = nn.Sequential(nn.Linear(32, 32))

    with pytest.raises(NotImplementedError, match="execute_stage_svd"):
        transformer.execute_stage_svd(model)

    with pytest.raises(NotImplementedError, match="execute_stage_prune"):
        transformer.execute_stage_prune(model)

    with pytest.raises(NotImplementedError, match="execute_stage_quantize"):
        transformer.execute_stage_quantize(model)
