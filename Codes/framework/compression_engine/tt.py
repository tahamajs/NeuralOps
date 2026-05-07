import torch.nn as nn
from framework.core.registry import COMPRESSOR_REGISTRY

@COMPRESSOR_REGISTRY.register("tt")
def tt_factory(model: nn.Module, comp_config) -> nn.Module:
    """
    Placeholder TT compressor (no actual compression).
    Replace with real implementation once TTLinearLayer is available.
    """
    tt_rank = getattr(comp_config, 'tt_rank', 16)
    print(f"[TT Placeholder] rank={tt_rank} – returning model unchanged")
    return model
