r"""
Native Hybrid Attention (NHA) Compressor.

Provides model-level transformation utilities for substituting conventional multi-head
self-attention blocks with Native Hybrid Attention (NHA) layers across transformer architectures.

Mathematical Mechanism:
    Given a transformer model $\mathcal{M}$ parameterized by weights $\Theta = \{\theta_{\text{attn}}^{(l)}, \theta_{\text{other}}^{(l)}\}_{l=1}^L$,
    the compressor maps:
    $$\mathcal{M} \mapsto \widetilde{\mathcal{M}} = \{\theta_{\text{NHA}}^{(l)}, \theta_{\text{other}}^{(l)}\}_{l=1}^L$$
    preserving feedforward, normalization, and embedding parameters while updating attention operator dispatch.

Computational Complexity:
    - Replacement dispatch: $\mathcal{O}(L)$ over $L$ transformer layers.
    - Resulting per-layer inference complexity is reduced from $\mathcal{O}(N^2 \cdot D)$ to the hybrid NHA complexity.
"""

from typing import Any, Optional, Tuple

import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils import replace_attention


@ATTENTION_REGISTRY.register("nha_compressor")
class NHACompressor:
    r"""Model compressor replacing standard attention layers with Native Hybrid Attention (NHA).

    Replaces every multi-head attention module within a transformer architecture with
    its NHA counterpart via the framework dynamic dispatch utility.

    Args:
        model: Target transformer neural network module (`nn.Module`).
        tokenizer: Associated huggingface or custom tokenizer instance.
        full_ratio: Fraction of full attention layers to retain if hybridizing ($0.0 \le \alpha \le 1.0$).
        device: Device string specification on which model tensors reside (e.g. `'cuda'`, `'cpu'`).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        full_ratio: float = 0.1,
        device: str = "cuda",
    ) -> None:
        self.model: nn.Module = model
        self.tokenizer: Any = tokenizer
        self.full_ratio: float = full_ratio
        self.device: str = device

    def compress(
        self, output_dir: Optional[str] = None
    ) -> Tuple[nn.Module, Any]:
        r"""Execute model-wide attention replacement and optionally persist weights.

        Args:
            output_dir: Optional directory path to save model weights and tokenizer configs.

        Returns:
            Tuple of `(compressed_model, tokenizer)`.
        """
        self.model = replace_attention(self.model, "nha")
        if output_dir is not None:
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(output_dir)
            if hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(output_dir)
        return self.model, self.tokenizer
