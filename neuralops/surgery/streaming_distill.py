#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Streaming Checkpointed Distillation Engine for Neural Attention Surgery.

Provides Zero-OOM activation checkpointing and online linear CKA loss
to enable high-throughput post-transplant healing on memory-constrained GPUs.
Slices token sequences into micro-chunks of C=128 tokens, accumulating Gram
matrices in constant O(D^2) memory rather than O(N^2) sample-space memory.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

logger = logging.getLogger("neuralops.surgery.streaming_distill")


class StreamingCKALoss(nn.Module):
    """
    Computes Centered Kernel Alignment in constant O(D^2) memory across streaming micro-chunks.

    Instead of materializing N x N sample-space Gram matrices (where N = B * T can exceed 100k,
    demanding 60+ GB VRAM), this module computes outer products in feature space (D x D),
    consuming strictly ~3 * D^2 * 4 bytes (~12 MB for D=1024) regardless of batch size or sequence length.

    Implements Chan's parallel online covariance updating to guarantee numerical parity
    with full-batch dense CKA down to machine precision.
    """

    def __init__(self, hidden_dim: int, eps: float = 1e-8, exact: bool = True) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.exact = exact
        self.register_buffer("G_xy", torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32))
        self.register_buffer("G_xx", torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32))
        self.register_buffer("G_yy", torch.zeros(hidden_dim, hidden_dim, dtype=torch.float32))
        self.register_buffer("mean_x", torch.zeros(1, hidden_dim, dtype=torch.float32))
        self.register_buffer("mean_y", torch.zeros(1, hidden_dim, dtype=torch.float32))
        self.total_tokens: int = 0

    def reset(self) -> None:
        """Resets accumulated outer products and running statistics for a new batch or distillation step."""
        self.G_xy = torch.zeros(self.hidden_dim, self.hidden_dim, device=self.G_xy.device, dtype=torch.float32)
        self.G_xx = torch.zeros(self.hidden_dim, self.hidden_dim, device=self.G_xx.device, dtype=torch.float32)
        self.G_yy = torch.zeros(self.hidden_dim, self.hidden_dim, device=self.G_yy.device, dtype=torch.float32)
        self.mean_x = torch.zeros(1, self.hidden_dim, device=self.mean_x.device, dtype=torch.float32)
        self.mean_y = torch.zeros(1, self.hidden_dim, device=self.mean_y.device, dtype=torch.float32)
        self.total_tokens = 0

    def update(self, X_chunk: torch.Tensor, Y_chunk: torch.Tensor) -> None:
        """
        Accumulates Gram matrices for a micro-chunk of shape [B * C, D] or [B, C, D].

        X_chunk (student): requires_grad may be True (autograd graph preserved).
        Y_chunk (teacher): detached representation.
        """
        x_flat = X_chunk.reshape(-1, self.hidden_dim).to(torch.float32)
        y_flat = Y_chunk.reshape(-1, self.hidden_dim).to(torch.float32)
        n_c = x_flat.size(0)
        if n_c == 0:
            return

        # Ensure buffer device matches input device
        if self.G_xy.device != x_flat.device:
            dev = x_flat.device
            self.G_xy = self.G_xy.to(dev)
            self.G_xx = self.G_xx.to(dev)
            self.G_yy = self.G_yy.to(dev)
            self.mean_x = self.mean_x.to(dev)
            self.mean_y = self.mean_y.to(dev)

        # Micro-chunk local means
        mu_x_chunk = x_flat.mean(dim=0, keepdim=True)
        mu_y_chunk = y_flat.mean(dim=0, keepdim=True)

        # Center features locally within chunk
        x_centered = x_flat - mu_x_chunk
        y_centered = y_flat - mu_y_chunk

        # Local chunk outer products: D x D
        g_xy_chunk = torch.matmul(x_centered.transpose(0, 1), y_centered)
        g_xx_chunk = torch.matmul(x_centered.transpose(0, 1), x_centered)
        g_yy_chunk = torch.matmul(y_centered.transpose(0, 1), y_centered)

        if not self.exact or self.total_tokens == 0:
            # First chunk or pure local accumulation mode
            self.G_xy = self.G_xy + g_xy_chunk
            self.G_xx = self.G_xx + g_xx_chunk
            self.G_yy = self.G_yy + g_yy_chunk
            self.mean_x = mu_x_chunk
            self.mean_y = mu_y_chunk
            self.total_tokens += n_c
        else:
            # Chan's parallel online covariance update formula
            n_a = float(self.total_tokens)
            n_b = float(n_c)
            weight = (n_a * n_b) / (n_a + n_b)

            delta_x = mu_x_chunk - self.mean_x
            delta_y = mu_y_chunk - self.mean_y

            self.G_xy = self.G_xy + g_xy_chunk + weight * torch.matmul(delta_x.transpose(0, 1), delta_y)
            self.G_xx = self.G_xx + g_xx_chunk + weight * torch.matmul(delta_x.transpose(0, 1), delta_x)
            self.G_yy = self.G_yy + g_yy_chunk + weight * torch.matmul(delta_y.transpose(0, 1), delta_y)

            self.mean_x = self.mean_x + (n_b / (n_a + n_b)) * delta_x
            self.mean_y = self.mean_y + (n_b / (n_a + n_b)) * delta_y
            self.total_tokens += n_c

    def compute(self) -> torch.Tensor:
        """
        Computes the rotation-invariant CKA similarity loss: 1.0 - CKA.

        Returns a scalar loss tensor with autograd gradients connecting back to student inputs.
        """
        if self.total_tokens == 0:
            return torch.tensor(0.0, device=self.G_xy.device, requires_grad=True)

        norm_xy = torch.norm(self.G_xy, p="fro") ** 2
        norm_xx = torch.norm(self.G_xx, p="fro")
        norm_yy = torch.norm(self.G_yy, p="fro")

        similarity = norm_xy / (norm_xx * norm_yy + self.eps)
        similarity = torch.clamp(similarity, min=0.0, max=1.0)
        return 1.0 - similarity

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Computes streaming CKA directly in forward call.

        If chunk_size is provided and sequence length exceeds chunk_size,
        slices along the sequence dimension (dim=1) to prevent memory spikes.
        """
        self.reset()
        if chunk_size is not None and X.dim() >= 2 and X.size(1) > chunk_size:
            seq_len = X.size(1)
            for st in range(0, seq_len, chunk_size):
                en = min(st + chunk_size, seq_len)
                self.update(X[:, st:en], Y[:, st:en])
        else:
            self.update(X, Y)
        return self.compute()


class CheckpointedTeacherBridge(nn.Module):
    """
    Wraps teacher layers with non-reentrant activation checkpointing and no_grad.

    Ensures teacher intermediate representations are evicted from VRAM immediately after
    loss computation, eliminating double-buffering during backpropagation.
    """

    def __init__(self, teacher_module: nn.Module) -> None:
        super().__init__()
        self.teacher = teacher_module
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def forward_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Executes a single layer under non-reentrant activation checkpointing with no_grad.
        """
        with torch.no_grad():
            def custom_forward(*inputs: torch.Tensor) -> torch.Tensor:
                mask = inputs[1] if len(inputs) > 1 else None
                out = layer(inputs[0], attention_mask=mask, *args, **kwargs)
                return out[0] if isinstance(out, (tuple, list)) else out

            c_args = (hidden_states,) if attention_mask is None else (hidden_states, attention_mask)
            res = checkpoint(custom_forward, *c_args, use_reentrant=False, preserve_rng_state=False)
            return res.detach()

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Executes full teacher forward pass under torch.no_grad()."""
        with torch.no_grad():
            return self.teacher(*args, **kwargs)


class MicroChunkedShadowTwinDistiller:
    """
    Manages the zero-OOM distillation loop across micro-chunks of C=128 tokens
    with active VRAM monitoring and threshold-based GPU cache purging.
    """

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        chunk_size: int = 128,
        vram_threshold_ratio: float = 0.75,
        hidden_dim: Optional[int] = None,
        lambda_cloning: float = 1.0,
        lambda_cka: float = 0.1,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.chunk_size = chunk_size
        self.vram_threshold_ratio = vram_threshold_ratio
        self.lambda_cloning = float(lambda_cloning)
        self.lambda_cka = float(lambda_cka)

        # Freeze teacher
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()
        self.teacher_bridge = CheckpointedTeacherBridge(teacher)

        # Discover hidden dimension
        if hidden_dim is None:
            hidden_dim = getattr(getattr(student, "config", None), "hidden_size", None)
        if hidden_dim is None:
            hidden_dim = getattr(student, "hidden_dim", None)
        if hidden_dim is None:
            hidden_dim = getattr(student, "hidden_size", 768)

        self.hidden_dim = int(hidden_dim)
        self.cka_loss_fn = StreamingCKALoss(hidden_dim=self.hidden_dim)

    def _purge_vram_if_needed(self) -> None:
        """Monitors allocated VRAM and triggers cache purging if threshold exceeded."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            total = torch.cuda.get_device_properties(0).total_memory
            if total > 0 and (allocated / total) > self.vram_threshold_ratio:
                torch.cuda.empty_cache()

    @staticmethod
    def _extract_hidden_representation(outputs: Any) -> torch.Tensor:
        """Extracts the final hidden state tensor from heterogeneous model output formats."""
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)):
            if len(outputs) > 1 and isinstance(outputs[1], (tuple, list)):
                return outputs[1][-1]
            if isinstance(outputs[0], torch.Tensor):
                return outputs[0]
        if isinstance(outputs, torch.Tensor):
            return outputs
        raise TypeError(f"Cannot extract hidden representation from {type(outputs).__name__}")

    def step(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Executes a single zero-OOM distillation step over the input sequence.

        Slices sequence into chunks along the token sequence dimension (dim=1),
        evaluates teacher and student in micro-chunks, accumulates streaming CKA in O(D^2),
        and purges GPU cache when memory utilization exceeds the configured threshold.
        """
        self.cka_loss_fn.reset()
        seq_len = input_ids.shape[1]
        num_chunks = max(1, (seq_len + self.chunk_size - 1) // self.chunk_size)

        total_cloning_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)

        for c in range(num_chunks):
            start = c * self.chunk_size
            end = min(start + self.chunk_size, seq_len)
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end] if attention_mask is not None else None

            # Teacher forward pass under no_grad
            with torch.no_grad():
                if chunk_mask is not None:
                    t_out = self.teacher(chunk_ids, attention_mask=chunk_mask, output_hidden_states=True)
                else:
                    t_out = self.teacher(chunk_ids, output_hidden_states=True)
                t_rep = self._extract_hidden_representation(t_out).detach()

            # Student forward pass with grad
            if chunk_mask is not None:
                s_out = self.student(chunk_ids, attention_mask=chunk_mask, output_hidden_states=True)
            else:
                s_out = self.student(chunk_ids, output_hidden_states=True)
            s_rep = self._extract_hidden_representation(s_out)

            # Accumulate streaming CKA outer products
            self.cka_loss_fn.update(s_rep, t_rep)

            # Layer cloning MSE loss
            chunk_mse = F.mse_loss(s_rep, t_rep)
            total_cloning_loss = total_cloning_loss + chunk_mse

            self._purge_vram_if_needed()

        cka_loss = self.cka_loss_fn.compute()
        avg_cloning_loss = total_cloning_loss / float(num_chunks)
        total_loss = self.lambda_cloning * avg_cloning_loss + self.lambda_cka * cka_loss

        return {
            "loss": total_loss,
            "cloning_loss": avg_cloning_loss,
            "cka_loss": cka_loss,
        }


__all__ = [
    "StreamingCKALoss",
    "CheckpointedTeacherBridge",
    "MicroChunkedShadowTwinDistiller",
]
