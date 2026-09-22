"""
UPI-Mamba / SelectiveMamba re-export module.
Provides direct access to SelectiveMamba (formerly UPIMambaBlock) and inject_selective_mamba.
"""

from framework.attentions.upi_stuffed_mamba import (
    SelectiveMamba,
    UPIMambaBlock,
    inject_selective_mamba,
    inject_upi_mamba,
)

__all__ = [
    "SelectiveMamba",
    "UPIMambaBlock",
    "inject_selective_mamba",
    "inject_upi_mamba",
]
