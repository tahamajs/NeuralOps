# framework/attentions/specialized/__init__.py
"""
Specialized attention mechanisms (AdaSplash, FlashAttention‑3, NHA, etc.).
All modules are automatically registered in ATTENTION_REGISTRY.
"""

# Import all modules to trigger registration
from . import adasplash_attention
from . import flash_attention3
from . import flash_attention3_fp8
from . import full_attention
from . import hamming_attention
from . import hla_attention
from . import infini_attention
from . import linformer_attention
from . import nha_attention
from . import vig_attention
from . import spectra_v2
from . import spectron
from . import vspec

# Optionally re‑export the classes for direct import
try:
    from .adasplash_attention import AdaSplashAttention
except ImportError:
    AdaSplashAttention = None

try:
    from .flash_attention3 import FlashAttention3
except ImportError:
    FlashAttention3 = None

try:
    from .flash_attention3_fp8 import FlashAttention3FP8
except ImportError:
    FlashAttention3FP8 = None

from .full_attention import FullAttention
from .hamming_attention import HammingAttention
from .hla_attention import HLAAttention
from .infini_attention import InfiniAttention
from .linformer_attention import LinformerAttention
from .nha_attention import NHAAttention
from .vig_attention import ViGAttention

__all__ = [
    "AdaSplashAttention",
    "FlashAttention3",
    "FlashAttention3FP8",
    "FullAttention",
    "HammingAttention",
    "HLAAttention",
    "InfiniAttention",
    "LinformerAttention",
    "NHAAttention",
    "ViGAttention",
]