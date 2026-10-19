# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Info Command

Displays system hardware status, registered attention modules, and compression engines.
"""

import sys
import platform


def cmd_info():
    import torch
    from neuralops.core.registry import (
        ATTENTION_REGISTRY, COMPRESSOR_REGISTRY,
    )
    import neuralops.attentions
    import neuralops.compression_engine

    print("neuralops")
    print("=" * 60)
    print(f"Python:      {sys.version.split()[0]}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"CUDA:        {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:         {torch.cuda.get_device_name(0)}")
    print(f"Platform:    {platform.platform()}")
    print()
    print(f"Attention:   {len(ATTENTION_REGISTRY.keys())} modules")
    print(f"Compressor:  {len(COMPRESSOR_REGISTRY.keys())} modules")
    return 0
