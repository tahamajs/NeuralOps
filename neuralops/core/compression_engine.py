#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Compression Pipeline Compatibility Shim

Backward-compatibility redirection interface routing legacy imports to the unified compression pipeline.
"""

import warnings

warnings.warn(
    "neuralops.core.compression_engine has moved to "
    "neuralops.core.compression_pipeline. Update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.core.compression_pipeline import *  # noqa: F401,F403
from neuralops.core.compression_pipeline import (  # noqa: F401
    CompressionPipeline,
    compress_model,
)
