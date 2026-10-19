# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CUDA Graph Decode Runner

CUDA Graph capture and replay engine for ultra-low latency autoregressive sequence decoding.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Dict, List, Union
import torch
import torch.nn as nn

from neuralops.models.cache import UnifiedHybridCache

logger = logging.getLogger("NeuralOps.CUDAGraph")


class CUDAGraphDecodeRunner:
    """
    CUDAGraphDecodeRunner encapsulates static graph capture for autoregressive 1-token decoding.

    Features:
    - Preallocates static input_ids buffer: shape [B, 1].
    - Preallocates static recurrent / hybrid states in UnifiedHybridCache.
    - Runs warmup steps to stabilize CUDA kernels and memory pools.
    - Captures decode graph via torch.cuda.CUDAGraph.
    - Dispatches instant execution via replay().
    """

    def __init__(
        self,
        model: nn.Module,
        batch_size: int = 1,
        device: Optional[torch.device | str] = None,
        warmup_steps: int = 3,
        max_batch_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        if max_batch_size is not None:
            batch_size = max_batch_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.warmup_steps = warmup_steps

        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_input_ids: Optional[torch.Tensor] = None
        self.static_cache: Optional[UnifiedHybridCache] = None
        self.static_logits: Optional[torch.Tensor] = None
        self.is_captured: bool = False

    def capture(self, sample_cache: Optional[UnifiedHybridCache] = None) -> None:
        """Warm up and capture the single-token decoding forward pass."""
        if not torch.cuda.is_available() or self.device.type != "cuda":
            logger.warning("CUDA is not available or device is not CUDA; skipping CUDA Graph capture.")
            self.is_captured = False
            return

        self.model.eval()

        self.static_input_ids = torch.zeros(
            (self.batch_size, 1), dtype=torch.long, device=self.device
        )

        if sample_cache is not None:
            self.static_cache = sample_cache
        else:
            num_layers = getattr(getattr(self.model, "config", None), "num_hidden_layers", 12)
            self.static_cache = UnifiedHybridCache(num_layers=num_layers)

        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(device=self.device))

        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(self.warmup_steps):
                out = self.model(
                    input_ids=self.static_input_ids,
                    past_key_values=self.static_cache,
                    use_cache=True,
                )
                if hasattr(out, "logits"):
                    _ = out.logits
                elif isinstance(out, (tuple, list)):
                    _ = out[0]

        torch.cuda.current_stream(device=self.device).wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.no_grad():
            out = self.model(
                input_ids=self.static_input_ids,
                past_key_values=self.static_cache,
                use_cache=True,
            )
            if hasattr(out, "logits"):
                self.static_logits = out.logits
            elif isinstance(out, (tuple, list)):
                self.static_logits = out[0]
            else:
                self.static_logits = out

        torch.cuda.current_stream(device=self.device).wait_stream(stream)
        self.is_captured = True
        logger.info(
            "Successfully captured CUDAGraph for model decode (batch_size=%d, device=%s)",
            self.batch_size,
            self.device,
        )

    def decode_step(self, token_id: torch.Tensor | int) -> torch.Tensor:
        """
        Execute 1 decode step using the captured CUDA graph.

        Args:
            token_id: Integer or 1D/2D tensor of shape [B, 1] or [B].

        Returns:
            Logits tensor of shape [B, 1, V].
        """
        if not self.is_captured or self.graph is None or self.static_input_ids is None:
            if isinstance(token_id, int):
                tok_tensor = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
            elif token_id.dim() == 1:
                tok_tensor = token_id.unsqueeze(1).to(self.device)
            else:
                tok_tensor = token_id.to(self.device)

            out = self.model(
                input_ids=tok_tensor,
                past_key_values=self.static_cache,
                use_cache=True,
            )
            if hasattr(out, "logits"):
                return out.logits
            elif isinstance(out, (tuple, list)):
                return out[0]
            return out

        if isinstance(token_id, int):
            self.static_input_ids.fill_(token_id)
        elif isinstance(token_id, torch.Tensor):
            if token_id.shape == self.static_input_ids.shape:
                self.static_input_ids.copy_(token_id)
            else:
                self.static_input_ids.copy_(token_id.view(self.batch_size, 1))

        self.graph.replay()
        return self.static_logits

    def step(self, token_id: Union[int, torch.Tensor]) -> torch.Tensor:
        """Alias for decode_step."""
        return self.decode_step(token_id)


__all__ = ["CUDAGraphDecodeRunner"]
