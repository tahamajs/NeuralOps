"""Simple State Space Model (SimpleSSM) - Deprecated wrapper aliasing SelectiveSSM.

Retained for backward-compatibility with ATTENTION_REGISTRY.
"""

from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.selective_ssm import SelectiveSSM

# Register SimpleSSM as an alias of SelectiveSSM for legacy compatibility
@ATTENTION_REGISTRY.register("simple_ssm", supports_causal=True)
class SimpleSSM(SelectiveSSM):
    """Legacy alias pointing directly to faithful SelectiveSSM."""
    pass
