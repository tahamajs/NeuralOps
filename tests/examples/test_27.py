#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Subprocess and unit regression tests for examples/27_huggingface_polymorphic_cache_bridge.py.
"""

from __future__ import annotations

import subprocess
import sys
import torch

from neuralops.models.cache import UnifiedHybridCache
from neuralops.surgery.hf_cache_bridge import (
    PolymorphicGenerationCacheBridge,
    validate_cache_format,
)


def test_27_runs_clean():
    """Verify example 27 executes cleanly via CLI subprocess with exit code 0."""
    proc = subprocess.run(
        [sys.executable, "examples/27_huggingface_polymorphic_cache_bridge.py", "--device", "cpu"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, f"Example 27 failed with stderr:\n{proc.stderr}"
    assert "[PASS] Overall Example 27 Status" in proc.stdout


def test_27_polymorphic_cache_contract_and_conversion():
    """Verify polymorphic cache conversion, seq length tracking, and legacy conversion."""
    cache = PolymorphicGenerationCacheBridge()
    validate_cache_format(cache)

    # Layer 0: Attention KV [2, 4, 8, 16]
    # Layer 1: Recurrent State [2, 4, 16, 16]
    cache.mark_recurrent_layer(1)

    k0 = torch.randn(2, 4, 8, 16)
    v0 = torch.randn(2, 4, 8, 16)
    cache.update(k0, v0, layer_idx=0)

    s1 = torch.randn(2, 4, 16, 16)
    cache.update_recurrent_state(1, s1, num_tokens=8)

    assert cache.get_seq_length(0) == 8
    assert cache.get_seq_length(1) == 8

    # Incremental update
    k0_next = torch.randn(2, 4, 1, 16)
    v0_next = torch.randn(2, 4, 1, 16)
    cache.update(k0_next, v0_next, layer_idx=0)

    s1_next = s1 + torch.randn(2, 4, 16, 16) * 0.05
    cache.update_recurrent_state(1, s1_next, num_tokens=1)

    assert cache.get_seq_length(0) == 9
    assert cache.get_seq_length(1) == 9

    # Verify to_legacy_cache and from_legacy_cache conversion
    legacy = cache.to_legacy_cache()
    assert len(legacy) >= 2

    # Verify interop with UnifiedHybridCache
    uhc = UnifiedHybridCache.from_legacy_cache(legacy)
    assert len(uhc) >= 2
    assert uhc.get_seq_length(0) == 9
