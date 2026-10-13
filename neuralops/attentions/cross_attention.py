"""
Backward-compatibility shim for neuralops.attentions.cross_attention.
"""
import warnings

warnings.warn(
    "neuralops.attentions.cross_attention is deprecated. "
    "Use neuralops.attentions.modern.cross_attention instead.",
    DeprecationWarning,
    stacklevel=2,
)

from neuralops.attentions.modern.cross_attention import *  # noqa: F401, F403
try:
    from neuralops.attentions.modern.cross_attention import CrossAttention
except ImportError:
    pass
