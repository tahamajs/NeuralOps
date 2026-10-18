"""
NeuralOps Autograd Core Modules.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.core.autograd.flash_attention import FlashAttentionFunction

__all__ = ["FlashAttentionFunction"]

# Auto-wired package members
from neuralops.core.autograd import flash_attention
from neuralops.core.autograd import fused_backward
