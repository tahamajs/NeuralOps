"""
Mamba Attention module alias.

Points to faithful MambaBlock architecture in framework.attentions.mamba_block.
"""

from framework.attentions.mamba_block import MambaBlock
from framework.attentions.mamba import Mamba

__all__ = ["MambaBlock", "Mamba"]
