#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Pipeline-Parallel Inference Engine for Multi-GPU Execution.

Splits transformer layers across available GPUs and processes input sequences
using asynchronous micro-batch pipelining to serve models exceeding single-GPU VRAM.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, List, Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from neuralops.distributed.context import DistributedContext
from neuralops.executor.layer_parallel import LayerParallelExecutor

logger = logging.getLogger(__name__)


class PipelineStage(nn.Module):
    """
    Represents an isolated partition of transformer blocks on a dedicated device.
    """

    def __init__(
        self,
        stage_id: int,
        device: torch.device,
        layers: Sequence[nn.Module],
        is_first_stage: bool,
        is_last_stage: bool,
        embed_tokens: Optional[nn.Module] = None,
        pos_embed: Optional[nn.Module] = None,
        rotary_emb: Optional[nn.Module] = None,
        norm: Optional[nn.Module] = None,
        lm_head: Optional[nn.Module] = None,
        config: Any = None,
    ) -> None:
        super().__init__()
        self.stage_id = stage_id
        self.device = device
        self.layers = nn.ModuleList(layers).to(device)
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
        self.embed_tokens = embed_tokens.to(device) if embed_tokens is not None else None
        self.pos_embed = pos_embed.to(device) if pos_embed is not None else None
        self.rotary_emb = copy.deepcopy(rotary_emb).to(device) if rotary_emb is not None else None
        self.norm = norm.to(device) if norm is not None else None
        self.lm_head = lm_head.to(device) if lm_head is not None else None
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute forward pass for the local stage.
        """
        if self.is_first_stage:
            input_ids = x.to(self.device)
            seq_len = input_ids.shape[1]
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            else:
                position_ids = position_ids.to(self.device)

            if self.embed_tokens is not None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_ids.float()

            if self.pos_embed is not None:
                hidden_states = hidden_states + self.pos_embed(position_ids)
        else:
            hidden_states = x.to(self.device)
            seq_len = hidden_states.shape[1]
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)
            else:
                position_ids = position_ids.to(self.device)

        position_embeddings = None
        if self.rotary_emb is not None:
            try:
                position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
            except Exception:
                position_embeddings = None

        layer_kwargs: dict[str, Any] = {}
        if position_embeddings is not None:
            layer_kwargs["position_embeddings"] = position_embeddings
        if position_ids is not None:
            layer_kwargs["position_ids"] = position_ids
        if attention_mask is not None:
            layer_kwargs["attention_mask"] = attention_mask.to(self.device)

        for layer in self.layers:
            out = layer(hidden_states, **layer_kwargs)
            hidden_states = out[0] if isinstance(out, tuple) else out

        if self.is_last_stage:
            if self.norm is not None:
                hidden_states = self.norm(hidden_states)
            if self.lm_head is not None:
                hidden_states = self.lm_head(hidden_states)

        return hidden_states


class PipelineParallelInference:
    """
    Executes micro-batch pipelined inference partitioned across GPUs.
    """

    def __init__(
        self,
        context: Optional[DistributedContext] = None,
        num_micro_batches: int = 4,
    ) -> None:
        self.context = context or DistributedContext()
        self.num_micro_batches = max(1, num_micro_batches)
        self.stages: List[PipelineStage] = []
        self.stage: Optional[PipelineStage] = None
        self.rank: int = self.context.rank if self.context.is_distributed() else 0
        if self.context.is_distributed():
            self.world_size = self.context.world_size
        elif self.context.available_gpus > 1:
            self.world_size = self.context.available_gpus
        else:
            self.world_size = 1

        self.model: Optional[nn.Module] = None
        self.owned_layers: List[nn.Module] = []
        self.streams: List[torch.cuda.Stream] = []
        self.is_single_gpu: bool = (self.world_size <= 1)

    @staticmethod
    def _extract_components(model: nn.Module) -> dict[str, Any]:
        """
        Extract transformer architectural building blocks from common model architectures.
        """
        layers = LayerParallelExecutor.find_layer_list(model)
        if layers is None:
            raise NotImplementedError(
                "PipelineParallelInference requires a model with transformer layers "
                "(e.g., LLaMA, GPT-2, Mistral, Qwen, BERT)."
            )

        embed_tokens = None
        if hasattr(model, "get_input_embeddings") and callable(model.get_input_embeddings):
            embed_tokens = model.get_input_embeddings()
        if embed_tokens is None and hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            embed_tokens = model.model.embed_tokens
        if embed_tokens is None and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            embed_tokens = model.transformer.wte
        if embed_tokens is None and hasattr(model, "embed_tokens"):
            embed_tokens = model.embed_tokens

        pos_embed = None
        if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
            pos_embed = model.transformer.wpe

        rotary_emb = None
        if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
            rotary_emb = model.model.rotary_emb
        elif hasattr(model, "rotary_emb"):
            rotary_emb = model.rotary_emb

        norm = None
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            norm = model.model.norm
        elif hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
            norm = model.transformer.ln_f
        elif hasattr(model, "norm"):
            norm = model.norm

        lm_head = None
        if hasattr(model, "lm_head"):
            lm_head = model.lm_head
        elif hasattr(model, "get_output_embeddings") and callable(model.get_output_embeddings):
            lm_head = model.get_output_embeddings()

        config = getattr(model, "config", None)

        return {
            "layers": layers,
            "embed_tokens": embed_tokens,
            "pos_embed": pos_embed,
            "rotary_emb": rotary_emb,
            "norm": norm,
            "lm_head": lm_head,
            "config": config,
        }

    def load_model(
        self,
        model_id: Union[str, nn.Module],
        dtype: torch.dtype = torch.bfloat16,
    ) -> nn.Module:
        """
        Load model weights, partition transformer blocks across pipeline stages,
        and place owned components on dedicated devices.
        """
        if isinstance(model_id, str):
            try:
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
            except Exception as e:
                raise RuntimeError(f"Failed to load model from model_id '{model_id}': {e}") from e
        elif isinstance(model_id, nn.Module):
            model = model_id
        else:
            raise TypeError(f"model_id must be a string or nn.Module, got {type(model_id)}")

        model.eval()
        components = self._extract_components(model)
        layers = list(components["layers"])
        num_layers = len(layers)

        if self.is_single_gpu:
            device = self.context.get_device()
            model.to(device)
            self.model = model
            self.owned_layers = layers
            return self.model

        world_size = self.world_size
        layer_partitions: List[List[int]] = []
        for s in range(world_size):
            start = s * num_layers // world_size
            end = (s + 1) * num_layers // world_size if s < world_size - 1 else num_layers
            layer_partitions.append(list(range(start, end)))

        if self.context.is_distributed():
            my_rank = self.rank
            my_indices = layer_partitions[my_rank]
            my_layers = [layers[i] for i in my_indices]
            self.owned_layers = my_layers
            device = self.context.get_device(my_rank)

            is_first = (my_rank == 0)
            is_last = (my_rank == world_size - 1)

            self.stage = PipelineStage(
                stage_id=my_rank,
                device=device,
                layers=my_layers,
                is_first_stage=is_first,
                is_last_stage=is_last,
                embed_tokens=components["embed_tokens"] if is_first else None,
                pos_embed=components["pos_embed"] if is_first else None,
                rotary_emb=components["rotary_emb"],
                norm=components["norm"] if is_last else None,
                lm_head=components["lm_head"] if is_last else None,
                config=components["config"],
            )
            self.stages = [self.stage]
            self.model = model
            return self.model

        self.stages = []
        self.streams = []
        for s in range(world_size):
            dev = torch.device(f"cuda:{s}")
            s_indices = layer_partitions[s]
            s_layers = [layers[i] for i in s_indices]
            is_first = (s == 0)
            is_last = (s == world_size - 1)

            stage_obj = PipelineStage(
                stage_id=s,
                device=dev,
                layers=s_layers,
                is_first_stage=is_first,
                is_last_stage=is_last,
                embed_tokens=components["embed_tokens"] if is_first else None,
                pos_embed=components["pos_embed"] if is_first else None,
                rotary_emb=components["rotary_emb"],
                norm=components["norm"] if is_last else None,
                lm_head=components["lm_head"] if is_last else None,
                config=components["config"],
            )
            self.stages.append(stage_obj)
            self.streams.append(torch.cuda.Stream(device=dev))

        self.owned_layers = layers
        self.model = model
        return self.model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Execute forward pass using micro-batch pipelining across ranks/GPUs.
        """
        if self.model is None and not self.stages:
            raise RuntimeError("Model has not been loaded. Call load_model() first.")

        if self.is_single_gpu:
            device = self.context.get_device()
            inp = input_ids.to(device)
            mask = attention_mask.to(device) if attention_mask is not None else None
            with torch.no_grad():
                out = self.model(inp, attention_mask=mask)
                return getattr(out, "logits", out)

        batch_size = input_ids.shape[0]
        actual_micro_batches = min(self.num_micro_batches, batch_size)
        micro_inputs = list(torch.chunk(input_ids, chunks=actual_micro_batches, dim=0))
        num_micro = len(micro_inputs)

        if attention_mask is not None:
            micro_masks = list(torch.chunk(attention_mask, chunks=actual_micro_batches, dim=0))
        else:
            micro_masks = [None] * num_micro

        if self.context.is_distributed():
            return self._forward_distributed(micro_inputs, micro_masks, input_ids.shape[1], input_ids.dtype)

        return self._forward_multigpu(micro_inputs, micro_masks)

    def _forward_multigpu(
        self,
        micro_inputs: List[torch.Tensor],
        micro_masks: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Execute event-based asynchronous micro-batch pipeline across CUDA devices.
        """
        num_stages = len(self.stages)
        num_micro = len(micro_inputs)
        total_steps = num_micro + num_stages - 1

        stage_intermediates: dict[int, dict[int, torch.Tensor]] = {
            s: {} for s in range(num_stages)
        }
        events: dict[int, dict[int, torch.cuda.Event]] = {
            s: {} for s in range(num_stages)
        }
        micro_logits: List[Optional[torch.Tensor]] = [None] * num_micro

        with torch.no_grad():
            for t in range(total_steps):
                for s in range(num_stages - 1, -1, -1):
                    m = t - s
                    if 0 <= m < num_micro:
                        stream = self.streams[s]
                        if s > 0:
                            stream.wait_event(events[s - 1][m])
                            inp = stage_intermediates[s - 1][m].to(stream.device, non_blocking=True)
                        else:
                            inp = micro_inputs[m].to(stream.device, non_blocking=True)

                        mask = micro_masks[m]
                        with torch.cuda.stream(stream):
                            out = self.stages[s](inp, attention_mask=mask)
                            if s == num_stages - 1:
                                micro_logits[m] = out
                            else:
                                stage_intermediates[s][m] = out

                            ev = torch.cuda.Event()
                            ev.record(stream)
                            events[s][m] = ev

            self.streams[-1].synchronize()

        valid_logits = [out for out in micro_logits if out is not None]
        return torch.cat(valid_logits, dim=0)

    def _forward_distributed(
        self,
        micro_inputs: List[torch.Tensor],
        micro_masks: List[Optional[torch.Tensor]],
        seq_len: int,
        input_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Execute rank-to-rank point-to-point distributed pipeline.
        """
        rank = self.rank
        world_size = self.world_size
        num_micro = len(micro_inputs)
        hidden_size = getattr(self.model.config, "hidden_size", 16) if hasattr(self.model, "config") else 16
        micro_logits: List[torch.Tensor] = []

        with torch.no_grad():
            for m in range(num_micro):
                mask = micro_masks[m]
                m_batch = micro_inputs[m].shape[0]

                if rank == 0:
                    inp = micro_inputs[m].to(self.stage.device)
                    out = self.stage(inp, attention_mask=mask)
                    dist.send(out.contiguous(), dst=1)
                elif 0 < rank < world_size - 1:
                    recv_buf = torch.empty((m_batch, seq_len, hidden_size), dtype=torch.float32, device=self.stage.device)
                    dist.recv(recv_buf, src=rank - 1)
                    out = self.stage(recv_buf, attention_mask=mask)
                    dist.send(out.contiguous(), dst=rank + 1)
                else:
                    recv_buf = torch.empty((m_batch, seq_len, hidden_size), dtype=torch.float32, device=self.stage.device)
                    dist.recv(recv_buf, src=rank - 1)
                    logits = self.stage(recv_buf, attention_mask=mask)
                    micro_logits.append(logits)

        if rank == world_size - 1:
            full_logits = torch.cat(micro_logits, dim=0)
        else:
            vocab_size = getattr(self.model.config, "vocab_size", 32000) if hasattr(self.model, "config") else 32000
            total_b = sum(inp.shape[0] for inp in micro_inputs)
            full_logits = torch.empty((total_b, seq_len, vocab_size), dtype=torch.float32, device=self.stage.device)

        dist.broadcast(full_logits, src=world_size - 1)
        return full_logits

    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Autoregressively generate tokens token-by-token using pipelined forward passes.
        """
        cur_ids = prompt_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(cur_ids, **kwargs)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur_ids = torch.cat([cur_ids, next_token.to(cur_ids.device)], dim=-1)
        return cur_ids


__all__ = ["PipelineStage", "PipelineParallelInference"]
