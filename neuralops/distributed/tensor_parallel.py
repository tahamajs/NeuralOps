# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tensor Parallelism Module

Implements tensor-parallel linear projections, column/row splitting, and all-reduce primitives.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.distributed as dist


class SimulatedTPGroup:
    """Metadata-only rank descriptor; it does not emulate collective operations."""

    def __init__(self, world_size: int = 2) -> None:
        if world_size <= 0:
            raise ValueError("world_size must be positive")
        self.world_size = world_size

    def get_world_size(self) -> int:
        return self.world_size


def _validate_tp_dimensions(tp_size: int, tp_rank: int) -> None:
    if not isinstance(tp_size, int) or isinstance(tp_size, bool):
        raise TypeError("tp_size must be an integer")
    if not isinstance(tp_rank, int) or isinstance(tp_rank, bool):
        raise TypeError("tp_rank must be an integer")
    if tp_size <= 0:
        raise ValueError(f"tp_size must be positive, got {tp_size}")
    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}")


def _require_tp_group(tp_size: int, process_group: Any, tp_rank: Optional[int] = None) -> None:
    """Reject collective layers that would otherwise silently return local-only results."""
    if isinstance(process_group, SimulatedTPGroup):
        raise TypeError("SimulatedTPGroup is metadata-only; pass a torch.distributed.ProcessGroup")
    if tp_size == 1:
        return
    if not dist.is_initialized():
        raise RuntimeError(
            "Tensor-parallel collectives require torch.distributed to be "
            "initialized; a SimulatedTPGroup does not perform collectives."
        )
    actual_size = dist.get_world_size(group=process_group)
    if actual_size != tp_size:
        raise ValueError(
            f"tp_size={tp_size} does not match process-group world size {actual_size}"
        )
    actual_rank = dist.get_rank(group=process_group)
    if tp_rank is not None and actual_rank != tp_rank:
        raise ValueError(
            f"tp_rank={tp_rank} does not match process-group rank {actual_rank}"
        )


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass-through on forward; All-Reduce on backward."""

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        group: Any = None,
        enabled: bool = False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.enabled = enabled
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        if ctx.enabled:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Tensor-parallel input-gradient reduction requires initialized "
                    "torch.distributed"
                )
            grad_output = grad_output.contiguous().clone()
            dist.all_reduce(grad_output, group=ctx.group)
        return grad_output, None, None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-Reduce on forward; Pass-through on backward."""

    @staticmethod
    def forward(
        ctx: Any,
        input: torch.Tensor,
        group: Any = None,
        enabled: bool = False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.enabled = enabled
        if enabled:
            if not dist.is_initialized():
                raise RuntimeError(
                    "Tensor-parallel all-reduce requires initialized torch.distributed"
                )
            output = input.clone()
            dist.all_reduce(output, group=group)
            return output
        return input

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        return grad_output, None, None


def copy_to_model_parallel_region(
    input: torch.Tensor, group: Any = None, *, enabled: Optional[bool] = None
) -> torch.Tensor:
    if isinstance(group, SimulatedTPGroup):
        raise TypeError("SimulatedTPGroup cannot be used for collective operations")
    if enabled is None:
        enabled = dist.is_initialized() and dist.get_world_size(group=group) > 1
    return _CopyToModelParallelRegion.apply(input, group, enabled)


def reduce_from_model_parallel_region(
    input: torch.Tensor, group: Any = None, *, enabled: Optional[bool] = None
) -> torch.Tensor:
    if isinstance(group, SimulatedTPGroup):
        raise TypeError("SimulatedTPGroup cannot be used for collective operations")
    if enabled is None:
        enabled = dist.is_initialized() and dist.get_world_size(group=group) > 1
    return _ReduceFromModelParallelRegion.apply(input, group, enabled)


class RankParallelSVDLinear(nn.Module):
    """
    Shards the inner SVD rank r across W workers (r_local = r / W).
    Since U V^T = sum_{p=0}^{W-1} U^(p) V^(p)T, each worker computes
    Y^(p) = (X V^(p)) U^(p)T locally with zero intermediate synchronization,
    followed by a single reduction and bias addition.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        bias: bool = True,
        process_group: Any = None,
    ) -> None:
        super().__init__()
        _validate_tp_dimensions(tp_size, tp_rank)
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("in_features, out_features, and rank must be positive")
        if rank % tp_size != 0:
            raise ValueError(f"rank {rank} must be divisible by tp_size {tp_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.local_rank = rank // tp_size
        self.process_group = process_group

        self.v_proj = nn.Linear(in_features, self.local_rank, bias=False)
        self.u_proj = nn.Linear(self.local_rank, out_features, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_tp_group(self.tp_size, self.process_group, self.tp_rank)
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        h = self.v_proj(x)
        out_local = self.u_proj(h)

        out = reduce_from_model_parallel_region(
            out_local, self.process_group, enabled=self.tp_size > 1
        )

        if self.bias is not None:
            out = out + self.bias
        return out


class RowParallelSVDLinear(nn.Module):
    """
    Accepts partitioned input X^(p) in R^{B x T x (d_in/W)}, shards V^(p)T in R^{r x (d_in/W)},
    sums in the compressed low-rank subspace R^{B x T x r} via All-Reduce (cutting communication
    by d_out / r), and finally applies local factor U in R^{d_out x r}.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        bias: bool = True,
        process_group: Any = None,
    ) -> None:
        super().__init__()
        _validate_tp_dimensions(tp_size, tp_rank)
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("in_features, out_features, and rank must be positive")
        if in_features % tp_size != 0:
            raise ValueError(f"in_features {in_features} must be divisible by tp_size {tp_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.local_in_features = in_features // tp_size
        self.process_group = process_group

        self.v_proj = nn.Linear(self.local_in_features, rank, bias=False)
        self.u_proj = nn.Linear(rank, out_features, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x_local: torch.Tensor) -> torch.Tensor:
        _require_tp_group(self.tp_size, self.process_group, self.tp_rank)
        if x_local.shape[-1] != self.local_in_features:
            raise ValueError(
                f"expected local input width {self.local_in_features}, "
                f"got {x_local.shape[-1]}"
            )
        bottleneck_local = self.v_proj(x_local)

        bottleneck = reduce_from_model_parallel_region(
            bottleneck_local, self.process_group, enabled=self.tp_size > 1
        )

        out = self.u_proj(bottleneck)
        if self.bias is not None:
            out = out + self.bias
        return out


class ColumnParallelSVDLinear(nn.Module):
    """
    Shards output features d_out across TP ranks (d_out_local = d_out / W).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        bias: bool = True,
        process_group: Any = None,
    ) -> None:
        super().__init__()
        _validate_tp_dimensions(tp_size, tp_rank)
        if in_features <= 0 or out_features <= 0 or rank <= 0:
            raise ValueError("in_features, out_features, and rank must be positive")
        if out_features % tp_size != 0:
            raise ValueError(f"out_features {out_features} must be divisible by tp_size {tp_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.local_out_features = out_features // tp_size
        self.process_group = process_group

        self.v_proj = nn.Linear(in_features, rank, bias=False)
        self.u_proj = nn.Linear(rank, self.local_out_features, bias=False)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.local_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        if self.tp_size > 1 and torch.is_grad_enabled() and x.requires_grad:
            _require_tp_group(self.tp_size, self.process_group, self.tp_rank)
        x_in = copy_to_model_parallel_region(x, self.process_group, enabled=self.tp_size > 1)
        h = self.v_proj(x_in)
        out = self.u_proj(h)
        if self.bias is not None:
            out = out + self.bias
        return out


class ColumnParallelLinear(nn.Module):
    """
    Standard Megatron-style ColumnParallelLinear.
    Splits output features d_out across TP workers (d_out_local = d_out / W).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        bias: bool = True,
        process_group: Any = None,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        _validate_tp_dimensions(tp_size, tp_rank)
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if out_features % tp_size != 0:
            raise ValueError(f"out_features {out_features} must be divisible by tp_size {tp_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.local_out_features = out_features // tp_size
        self.process_group = process_group
        self.gather_output = gather_output

        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.local_out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gather_output and self.tp_size > 1:
            _require_tp_group(self.tp_size, self.process_group, self.tp_rank)
        elif self.tp_size > 1 and torch.is_grad_enabled() and x.requires_grad:
            _require_tp_group(self.tp_size, self.process_group, self.tp_rank)
        if x.shape[-1] != self.in_features:
            raise ValueError(f"expected input width {self.in_features}, got {x.shape[-1]}")
        input_parallel = copy_to_model_parallel_region(
            x,
            self.process_group,
            enabled=self.tp_size > 1 and torch.is_grad_enabled() and x.requires_grad,
        )
        output_parallel = torch.matmul(input_parallel, self.weight.t())
        if self.bias is not None:
            output_parallel = output_parallel + self.bias
        if self.gather_output and self.tp_size > 1:
            from torch.distributed.nn.functional import all_gather

            output_parallel = torch.cat(
                all_gather(output_parallel, group=self.process_group), dim=-1
            )
        return output_parallel


class RowParallelLinear(nn.Module):
    """
    Standard Megatron-style RowParallelLinear.
    Splits input features d_in across TP workers (d_in_local = d_in / W).
    Performs local GEMM and an All-Reduce sum across the TP group.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        bias: bool = True,
        input_is_parallel: bool = True,
        process_group: Any = None,
    ) -> None:
        super().__init__()
        _validate_tp_dimensions(tp_size, tp_rank)
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        if in_features % tp_size != 0:
            raise ValueError(f"in_features {in_features} must be divisible by tp_size {tp_size}")
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.local_in_features = in_features // tp_size
        self.input_is_parallel = input_is_parallel
        self.process_group = process_group

        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _require_tp_group(self.tp_size, self.process_group, self.tp_rank)
        if self.input_is_parallel:
            if x.shape[-1] != self.local_in_features:
                raise ValueError(
                    f"expected local input width {self.local_in_features}, "
                    f"got {x.shape[-1]}"
                )
        elif x.shape[-1] != self.in_features:
            raise ValueError(f"expected full input width {self.in_features}, got {x.shape[-1]}")
        if not self.input_is_parallel and self.tp_size > 1:
            start_idx = self.tp_rank * self.local_in_features
            end_idx = start_idx + self.local_in_features
            x_local = x[..., start_idx:end_idx]
        else:
            x_local = x

        output_parallel = torch.matmul(x_local, self.weight.t())
        output_ = reduce_from_model_parallel_region(
            output_parallel, self.process_group, enabled=self.tp_size > 1
        )
        if self.bias is not None:
            output_ = output_ + self.bias
        return output_


class TensorParallelRecurrentAttention(nn.Module):
    """
    Partitions attention heads H across W TP workers (H_local = H / W).
    Query, Key, Value and Gate projections are column-sharded across heads.
    Each TP worker maintains only its local heads' recurrent state. The causal
    token scan is local to each rank; the final output projection is row-parallel
    and performs one all-reduce across the tensor-parallel group. Projection
    weights must be initialized from the matching rank-local slices of a base model.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        tp_size: int = 1,
        tp_rank: int = 0,
        process_group: Any = None,
    ) -> None:
        super().__init__()
        _validate_tp_dimensions(tp_size, tp_rank)
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if num_heads % tp_size != 0:
            raise ValueError(f"num_heads {num_heads} must be divisible by tp_size {tp_size}")
        if head_dim is not None and head_dim <= 0:
            raise ValueError("head_dim must be positive when provided")
        if head_dim is None and embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads when head_dim is omitted")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.local_heads = num_heads // tp_size
        self.local_dim = self.local_heads * self.head_dim
        self.process_group = process_group

        self.q_proj = nn.Linear(embed_dim, self.local_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.local_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.local_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, self.local_dim, bias=False)

        self.out_proj = nn.Linear(self.local_dim, embed_dim, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _require_tp_group(self.tp_size, self.process_group, self.tp_rank)
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.embed_dim:
            raise ValueError(f"hidden_states must have shape [B, T, {self.embed_dim}]")
        B, T, _ = hidden_states.shape
        x_in = copy_to_model_parallel_region(
            hidden_states, self.process_group, enabled=self.tp_size > 1
        )

        q = self.q_proj(x_in).view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_in).view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_in).view(B, T, self.local_heads, self.head_dim).transpose(1, 2)
        g = (
            torch.sigmoid(self.g_proj(x_in))
            .view(B, T, self.local_heads, self.head_dim)
            .transpose(1, 2)
        )

        scale = 1.0 / (self.head_dim**0.5)
        accumulator_dtype = (
            torch.float32 if k.dtype in (torch.float16, torch.bfloat16) else k.dtype
        )
        q_acc = q.to(accumulator_dtype)
        k_acc = k.to(accumulator_dtype)
        v_acc = v.to(accumulator_dtype)
        g_acc = g.to(accumulator_dtype)
        expected_state_shape = (B, self.local_heads, self.head_dim, self.head_dim)
        if state is None:
            state = torch.zeros(
                expected_state_shape, dtype=accumulator_dtype, device=k.device
            )
        elif tuple(state.shape) != expected_state_shape:
            raise ValueError(
                f"state must have shape {expected_state_shape}, got {tuple(state.shape)}"
            )
        else:
            state = state.to(accumulator_dtype)

        outputs = []
        for token_index in range(T):
            key_t = k_acc[:, :, token_index, :].unsqueeze(-1)
            value_t = (
                v_acc[:, :, token_index, :] * g_acc[:, :, token_index, :]
            ).unsqueeze(-2)
            state = state + torch.matmul(key_t, value_t)
            query_t = q_acc[:, :, token_index, :].unsqueeze(-2) * scale
            outputs.append(torch.matmul(query_t, state))
        if outputs:
            out_local = torch.cat(outputs, dim=-2)
        else:
            out_local = q.new_empty((B, self.local_heads, 0, self.head_dim))
        out_local = (
            out_local.transpose(1, 2)
            .contiguous()
            .view(B, T, self.local_dim)
            .to(self.out_proj.weight.dtype)
        )

        out_reduced = reduce_from_model_parallel_region(
            self.out_proj(out_local), self.process_group, enabled=self.tp_size > 1
        )
        return out_reduced, state


def shard_model_tensor_parallel(
    model: nn.Module,
    tp_size: int = 2,
    tp_rank: int = 0,
    process_group: Any = None,
) -> nn.Module:
    """Placeholder retained for API compatibility; automatic module sharding is unsupported."""
    _validate_tp_dimensions(tp_size, tp_rank)
    raise NotImplementedError(
        "Automatic model sharding is not implemented. Replace projections with "
        "ColumnParallelLinear/RowParallelLinear explicitly and load rank-local weights."
    )


__all__ = [
    "SimulatedTPGroup",
    "ColumnParallelLinear",
    "RowParallelLinear",
    "RankParallelSVDLinear",
    "RowParallelSVDLinear",
    "ColumnParallelSVDLinear",
    "TensorParallelRecurrentAttention",
    "shard_model_tensor_parallel",
    "copy_to_model_parallel_region",
    "reduce_from_model_parallel_region",
]
