"""
Orchestrates the application of one or more compressors to a model.
Reads a compression configuration and dispatches work to registered compressors.
"""

import torch.nn as nn
from typing import Optional, List, Union, Dict, Any
from .registry import COMPRESSOR_REGISTRY
from .base import BaseCompressor
from .utils import get_device, set_seed


class CompressionPipeline:
    """
    Apply a sequence of compression steps to a model.

    Usage:
        config = {
            'compressors': [
                {'method': 'a3', 'a3_rank': 32},
                {'method': 'kmeans', 'kmeans_clusters': 256},
            ],
            'tokenizer_attached': True,   # ensure model.tokenizer exists
            'seed': 42,
        }
        pipe = CompressionPipeline(model, config)
        compressed_model = pipe.run()
    """

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        tokenizer=None,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        # Attach tokenizer to model if provided (many compressors expect it)
        if tokenizer is not None:
            self.model.tokenizer = tokenizer

    def run(self) -> nn.Module:
        compressors_cfg = self.config.get('compressors', [])
        if not compressors_cfg:
            print("No compressors specified. Returning original model.")
            return self.model

        # Set random seed for reproducibility
        seed = self.config.get('seed')
        if seed is not None:
            set_seed(seed)

        for i, comp_cfg in enumerate(compressors_cfg):
            method = comp_cfg.pop('method')
            print(f"\n🔧 Step {i+1}: Applying compressor '{method}'")
            compressor_cls = COMPRESSOR_REGISTRY.resolve(method)

            # Merge with global config (optional)
            merged_cfg = {**self.config, **comp_cfg}
            # Remove the 'compressors' key if present to avoid recursion
            merged_cfg.pop('compressors', None)

            # Instantiate and compress
            compressor: BaseCompressor = compressor_cls(
                self.model, **merged_cfg
            )
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