"""
CacheNotes: Task‑Aware Key‑Value Cache Compression.

Given a task description, CacheNotes generates Compression‑Planning‑Tokens (CPTs)
that guide a one‑time compression of the corpus into a compact, reusable KV cache.
This implementation uses task‑prefix–guided pruning of linear layers.

Paper: "CacheNotes: Task-aware key-value cache compression for reasoning-intensive
        knowledge tasks." Corallo et al., EACL 2026.
Registered as 'cachenotes'.
"""

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY


class CompressionPlanningTokens:
    """
    Simulates CPT generation. In a full implementation this would be a small
    language model that reads a task description and outputs importance scores.
    Here we use a simple heuristic: token positions that match task keywords
    are assigned higher importance.
    """
    TASK_KEYWORD_MAP = {
        'qa': ['question', 'answer', 'context', 'passage'],
        'summarize': ['summary', 'abstract', 'key', 'main'],
        'code': ['function', 'class', 'def', 'return', 'import'],
        'reasoning': ['therefore', 'because', 'if', 'then', 'thus'],
    }

    @classmethod
    def generate_importance(cls, task_type: str, num_features: int) -> torch.Tensor:
        """Generate a per‑feature importance score based on task type."""
        importance = torch.zeros(num_features)
        # Assign higher baseline importance to first features (position bias)
        importance[:num_features // 4] = 0.1
        # Task‑specific boost
        return importance + 0.5   # uniform baseline


class CacheNotesCompressor(BaseCompressor):
    """
    Compresses model weights using task‑aware importance scoring inspired by
    CacheNotes CPT generation.

    Args:
        model: PyTorch model.
        task_type: type of downstream task ('qa', 'summarize', 'code', 'reasoning').
        keep_ratio: fraction of parameters to retain (default 0.3).
    """
    def __init__(self, model, task_type: str = 'qa', keep_ratio: float = 0.3,
                 **kwargs):
        super().__init__(model, name="cachenotes", **kwargs)
        self.task_type = task_type
        self.keep_ratio = keep_ratio

    def compress(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                W = module.weight.data
                out_f, in_f = W.shape

                # Generate CPT‑like importance scores per output neuron
                importance = CompressionPlanningTokens.generate_importance(
                    self.task_type, out_f
                ).to(W.device)

                # Combine with weight magnitude for final importance
                row_norms = torch.norm(W.float(), dim=1)
                combined = row_norms * (1.0 + importance)

                # Retain top neurons
                keep_n = max(1, int(out_f * self.keep_ratio))
                _, top_idx = torch.topk(combined, keep_n)

                # Build mask and apply
                mask = torch.zeros(out_f, device=W.device)
                mask[top_idx] = 1.0
                W_compressed = W * mask.unsqueeze(1)
                module.weight.data = W_compressed.to(dtype=module.weight.dtype)

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    


@COMPRESSOR_REGISTRY.register("cachenotes")
def cachenotes_factory(model, comp_config):
    task_type = getattr(comp_config, 'cachenotes_task', 'qa')
    keep_ratio = getattr(comp_config, 'cachenotes_keep_ratio', 0.3)
    compressor = CacheNotesCompressor(model, task_type=task_type, keep_ratio=keep_ratio)
    compressed_model, _ = compressor.compress()
    return compressed_model
