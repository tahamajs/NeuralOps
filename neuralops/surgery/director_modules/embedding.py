#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Embedding and Parameter Utility Module.

Provides helpers for inspecting, verifying, remediating tied embeddings (Law 8),
and managing distributed gathered parameter contexts (Law 9).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger("neuralops.surgery.director.embedding")


def get_input_embedding_module(model: nn.Module) -> Optional[nn.Module]:
    """Retrieves the input token embedding module across diverse architectures."""
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
            if emb is not None:
                return emb
        except Exception:
            pass

    for candidate in ("embed_tokens", "word_embeddings", "wte", "embeddings", "shared"):
        if hasattr(model, candidate):
            mod = getattr(model, candidate)
            if hasattr(mod, "weight"):
                return mod
            if hasattr(mod, "word_embeddings"):
                return mod.word_embeddings

    # Check child model if wrapped (e.g. model.model, model.transformer, model.bert)
    for sub in ("model", "transformer", "bert", "encoder"):
        if hasattr(model, sub):
            sub_mod = getattr(model, sub)
            sub_emb = get_input_embedding_module(sub_mod)
            if sub_emb is not None:
                return sub_emb

    return None


_get_input_embedding_module = get_input_embedding_module


def get_output_embedding_module(model: nn.Module) -> Optional[nn.Module]:
    """Retrieves the output projection / lm_head module across diverse architectures."""
    if hasattr(model, "get_output_embeddings"):
        try:
            head = model.get_output_embeddings()
            if head is not None:
                return head
        except Exception:
            pass

    for candidate in ("lm_head", "output_projection", "cls", "head"):
        if hasattr(model, candidate):
            mod = getattr(model, candidate)
            if hasattr(mod, "weight"):
                return mod
            if hasattr(mod, "predictions"):
                pred = mod.predictions
                if hasattr(pred, "weight"):
                    return pred
                if hasattr(pred, "decoder") and hasattr(pred.decoder, "weight"):
                    return pred.decoder

    return None


_get_output_embedding_module = get_output_embedding_module


def verify_tied_embeddings(model: nn.Module) -> bool:
    """
    Law 8: Verifies input and output token embeddings share the exact same weight pointer
    and underlying storage:
    id(out_head.weight) == id(inp_emb.weight) and
    out_head.weight.data_ptr() == inp_emb.weight.data_ptr().
    """
    inp_emb = get_input_embedding_module(model)
    out_head = get_output_embedding_module(model)

    if inp_emb is None or out_head is None:
        return True

    inp_w = getattr(inp_emb, "weight", None)
    out_w = getattr(out_head, "weight", None)

    if inp_w is None or out_w is None:
        return True

    is_same_obj = id(inp_w) == id(out_w)
    is_same_ptr = False
    try:
        is_same_ptr = inp_w.data_ptr() == out_w.data_ptr()
    except Exception:
        pass

    return is_same_obj and is_same_ptr


def remediate_tied_embeddings(model: nn.Module) -> bool:
    """
    Law 8 Remediation: Re-links lm_head.weight to input embedding weight if un-tied:
    model.lm_head.weight = model.get_input_embeddings().weight.
    Guarantees shared pointers and data_ptr() remain synchronized without memory corruption.
    """
    inp_emb = get_input_embedding_module(model)
    out_head = get_output_embedding_module(model)

    if inp_emb is not None and out_head is not None and hasattr(inp_emb, "weight"):
        out_head.weight = inp_emb.weight
        logger.info("Law 8: Re-tied output embedding weight to input embedding weight.")
        return True

    return False


_remediate_tied_embeddings = remediate_tied_embeddings
re_tie_embeddings = remediate_tied_embeddings


class GatheredParametersContext:
    """
    Law 9 (Distributed Sharding Invariant):
    Context manager handling deepspeed.zero.GatheredParameters or PyTorch FSDP.summon_full_params
    to ensure weights are safely gathered across GPU ranks before extraction or weight transfer.
    """

    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self._ctx: Any = None

    def __enter__(self) -> None:
        try:
            from deepspeed import zero

            if hasattr(zero, "GatheredParameters"):
                self._ctx = zero.GatheredParameters(self.module.parameters(), modifier_rank=0)
                self._ctx.__enter__()
                return
        except Exception:
            pass

        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            if isinstance(self.module, FSDP):
                self._ctx = FSDP.summon_full_params(self.module)
                self._ctx.__enter__()
                return
        except Exception:
            pass

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._ctx is not None:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
