"""
Compression pipeline – orchestrates multiple compressors sequentially.
"""

import torch.nn as nn
from typing import Dict, Any, Optional
from .registry import COMPRESSOR_REGISTRY
from .utils import set_seed


class CompressionPipeline:
    """Apply a sequence of compression steps to a model."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        if tokenizer is not None:
            self.model.tokenizer = tokenizer

    def run(self) -> nn.Module:
        compressors_cfg = self.config.get('compressors', [])
        if not compressors_cfg:
            print("No compressors specified. Returning original model.")
            return self.model

        seed = self.config.get('seed')
        if seed is not None:
            set_seed(seed)

        for i, comp_cfg in enumerate(compressors_cfg):
            method = comp_cfg.pop('method')
            print(f"\n🔧 Step {i+1}: Applying compressor '{method}'")
            compressor_cls = COMPRESSOR_REGISTRY.resolve(method)
            merged_cfg = {**self.config, **comp_cfg}
            merged_cfg.pop('compressors', None)
            compressor = compressor_cls(self.model, **merged_cfg)
            self.model, stats = compressor.compress()
            if stats:
                print(compressor.summary())

        return self.model


def compress_model(
    model: nn.Module,
    config: Dict[str, Any],
    tokenizer=None,
) -> nn.Module:
    """Convenience function to compress a model with a configuration dict."""
    pipeline = CompressionPipeline(model, config, tokenizer)
    return pipeline.run()