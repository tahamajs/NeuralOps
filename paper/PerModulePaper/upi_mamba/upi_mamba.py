"""
UPI-Mamba re-export module.
Provides direct access to UPIMambaBlock and inject_upi_mamba.
"""

from framework.attentions.upi_stuffed_mamba import UPIMambaBlock, inject_upi_mamba

__all__ = ["UPIMambaBlock", "inject_upi_mamba"]
