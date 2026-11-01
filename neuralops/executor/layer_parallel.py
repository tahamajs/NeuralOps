#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Layer-Parallel Transformation Executor for Multi-GPU Model Surgery.
"""

from __future__ import annotations

import concurrent.futures
import copy
import logging
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from neuralops.distributed.context import DistributedContext

logger = logging.getLogger(__name__)


class LayerParallelExecutor:
    """
    Executes model surgery and layer-wise transformations in parallel across available GPUs.

    Partitions transformer layer blocks across GPUs, applies transformations independently,
    and synchronizes and merges the results back into the model container.
    Falls back gracefully to single-GPU or CPU execution if only 1 device is available or
    if the architecture does not expose a standard layer list.
    """

    def __init__(self, context: Optional[DistributedContext] = None) -> None:
        self.context = context or DistributedContext()
        if self.context.is_distributed():
            self.world_size = self.context.world_size
        elif self.context.available_gpus > 1:
            self.world_size = self.context.available_gpus
        else:
            self.world_size = 1

    @staticmethod
    def find_layer_list(model: nn.Module) -> Optional[Sequence[nn.Module]]:
        """
        Locate transformer block layer stack in standard architectures.

        Supports:
        - LLaMA, Qwen, Mistral (`model.model.layers` or `model.layers`)
        - GPT-2, Bloom (`model.transformer.h` or `model.h`)
        - BERT, RoBERTa (`model.encoder.layer` or `model.bert.encoder.layer`)
        - DistilBERT (`model.transformer.layer` or `model.distilbert.transformer.layer`)

        Returns:
            nn.ModuleList or list container if found, otherwise None (non-layer-stackable model).
        """
        # 1. model.model.layers (LLaMA, Qwen, Mistral)
        if hasattr(model, "model") and hasattr(model.model, "layers") and isinstance(model.model.layers, (nn.ModuleList, list)):
            return model.model.layers
        if hasattr(model, "layers") and isinstance(model.layers, (nn.ModuleList, list)):
            return model.layers

        # 2. model.transformer.h (GPT-2, Bloom)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h") and isinstance(model.transformer.h, (nn.ModuleList, list)):
            return model.transformer.h
        if hasattr(model, "h") and isinstance(model.h, (nn.ModuleList, list)):
            return model.h

        # 3. model.encoder.layer (BERT, RoBERTa)
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer") and isinstance(model.encoder.layer, (nn.ModuleList, list)):
            return model.encoder.layer
        if (
            hasattr(model, "bert")
            and hasattr(model.bert, "encoder")
            and hasattr(model.bert.encoder, "layer")
            and isinstance(model.bert.encoder.layer, (nn.ModuleList, list))
        ):
            return model.bert.encoder.layer

        # 4. model.transformer.layer (DistilBERT)
        if hasattr(model, "transformer") and hasattr(model.transformer, "layer") and isinstance(model.transformer.layer, (nn.ModuleList, list)):
            return model.transformer.layer
        if (
            hasattr(model, "distilbert")
            and hasattr(model.distilbert, "transformer")
            and hasattr(model.distilbert.transformer, "layer")
            and isinstance(model.distilbert.transformer.layer, (nn.ModuleList, list))
        ):
            return model.distilbert.transformer.layer

        return None

    def transform_layers(
        self,
        model: nn.Module,
        layer_indices: Optional[List[int]] = None,
        op_fn: Optional[Callable[[nn.Module], nn.Module]] = None,
    ) -> nn.Module:
        """
        Apply an operation function across specified layers using layer parallelism.

        Parameters
        ----------
        model : nn.Module
            Model to transform.
        layer_indices : list[int], optional
            Indices of layers to transform. If None, transforms all layers.
        op_fn : Callable[[nn.Module], nn.Module], optional
            Transformation function applied to each layer block.

        Returns
        -------
        nn.Module
            Transformed model.
        """
        layers = self.find_layer_list(model)
        if layers is None:
            logger.warning("find_layer_list returned None (non-layer-stackable model). Falling back to single-GPU transformation.")
            if op_fn is not None:
                return op_fn(model)
            return model

        total_layers = len(layers)
        target_indices = layer_indices if layer_indices is not None else list(range(total_layers))

        if op_fn is None or not target_indices:
            return model

        target_dev = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")

        # Fallback to single-device if only 1 worker/GPU available
        if self.world_size <= 1 or not torch.cuda.is_available() or self.context.available_gpus <= 1:
            for idx in target_indices:
                layers[idx] = op_fn(layers[idx])
            return model

        # Multi-GPU execution
        num_gpus = self.context.available_gpus if not self.context.is_distributed() else self.world_size

        def _worker(r: int) -> List[Tuple[int, nn.Module]]:
            dev = torch.device(f"cuda:{r}")
            torch.cuda.set_device(dev)
            my_indices = target_indices[r::num_gpus]
            out: List[Tuple[int, nn.Module]] = []
            for idx in my_indices:
                orig_layer = layers[idx].to(dev)
                transformed = op_fn(orig_layer)
                torch.cuda.synchronize(dev)
                out.append((idx, transformed))
            return out

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            worker_results = list(executor.map(_worker, range(num_gpus)))

        for wr in worker_results:
            for idx, trans_layer in wr:
                layers[idx] = trans_layer.to(target_dev)

        self.context.barrier()
        return model

    def transplant(
        self,
        model: nn.Module,
        target_attention: str = "full_attention",
        layer_budget: Union[str, List[int], int] = "all",
        **kwargs: Any,
    ) -> nn.Module:
        """
        Transplant attention mechanism across model layers using layer parallelism.

        Parameters
        ----------
        model : nn.Module
            Target model.
        target_attention : str, optional
            Registered attention mechanism name, by default "full_attention".
        layer_budget : Union[str, List[int], int], optional
            Layers to transplant ("all", list of layer indices, or count int), by default "all".
        **kwargs : Any
            Additional keyword arguments passed to replace_attention.

        Returns
        -------
        nn.Module
            Surgically modified model.
        """
        from neuralops.api import replace_attention

        layers = self.find_layer_list(model)
        if layers is None:
            logger.warning("find_layer_list returned None (non-layer-stackable model). Falling back to single-device transplant.")
            return replace_attention(model, attention=target_attention, **kwargs)

        num_layers = len(layers)
        if layer_budget == "all":
            indices = list(range(num_layers))
        elif isinstance(layer_budget, list):
            indices = layer_budget
        elif isinstance(layer_budget, int):
            indices = list(range(min(layer_budget, num_layers)))
        else:
            indices = list(range(num_layers))

        target_dev = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        model_cfg = getattr(model, "config", None)
        seed = kwargs.pop("seed", 42)

        if self.world_size <= 1 or not torch.cuda.is_available() or self.context.available_gpus <= 1:
            for idx in indices:
                torch.manual_seed(seed + idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + idx)
                lyr = layers[idx]
                if model_cfg is not None and not hasattr(lyr, "config"):
                    lyr.config = model_cfg
                layers[idx] = replace_attention(lyr, attention=target_attention, **kwargs)
            return model

        num_gpus = self.context.available_gpus if not self.context.is_distributed() else self.world_size

        def _worker(r: int) -> List[Tuple[int, nn.Module]]:
            dev = torch.device(f"cuda:{r}")
            torch.cuda.set_device(dev)
            my_indices = indices[r::num_gpus]
            out: List[Tuple[int, nn.Module]] = []
            for idx in my_indices:
                orig_layer = layers[idx].to(dev)
                if model_cfg is not None and not hasattr(orig_layer, "config"):
                    orig_layer.config = model_cfg
                torch.manual_seed(seed + idx)
                torch.cuda.manual_seed_all(seed + idx)
                transformed = replace_attention(orig_layer, attention=target_attention, **kwargs)
                torch.cuda.synchronize(dev)
                out.append((idx, transformed))
            return out

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            worker_results = list(executor.map(_worker, range(num_gpus)))

        for wr in worker_results:
            for idx, trans_layer in wr:
                layers[idx] = trans_layer.to(target_dev)

        self.context.barrier()
        return model

    def compress(
        self,
        model: nn.Module,
        method: str = "svd",
        target_ratio: float = 0.5,
        layer_budget: Union[str, List[int], int] = "all",
        **kwargs: Any,
    ) -> nn.Module:
        """
        Compress model layers using layer parallelism.

        Parameters
        ----------
        model : nn.Module
            Target model.
        method : str, optional
            Registered compressor name, by default "svd".
        target_ratio : float, optional
            Target compression ratio (or parameter ratio), by default 0.5.
        layer_budget : Union[str, List[int], int], optional
            Layers to compress ("all", list of indices, or count int), by default "all".
        **kwargs : Any
            Additional keyword arguments passed to compress.

        Returns
        -------
        nn.Module
            Compressed model.
        """
        from neuralops.api import compress

        layers = self.find_layer_list(model)
        if layers is None:
            logger.warning("find_layer_list returned None (non-layer-stackable model). Falling back to single-device compress.")
            return compress(model, method=method, **kwargs)

        num_layers = len(layers)
        if layer_budget == "all":
            indices = list(range(num_layers))
        elif isinstance(layer_budget, list):
            indices = layer_budget
        elif isinstance(layer_budget, int):
            indices = list(range(min(layer_budget, num_layers)))
        else:
            indices = list(range(num_layers))

        target_dev = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        model_cfg = getattr(model, "config", None)
        seed = kwargs.pop("seed", 42)

        if self.world_size <= 1 or not torch.cuda.is_available() or self.context.available_gpus <= 1:
            for idx in indices:
                torch.manual_seed(seed + idx)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + idx)
                lyr = layers[idx]
                if model_cfg is not None and not hasattr(lyr, "config"):
                    lyr.config = model_cfg
                layers[idx] = compress(lyr, method=method, **kwargs)
            return model

        num_gpus = self.context.available_gpus if not self.context.is_distributed() else self.world_size

        def _worker(r: int) -> List[Tuple[int, nn.Module]]:
            dev = torch.device(f"cuda:{r}")
            torch.cuda.set_device(dev)
            my_indices = indices[r::num_gpus]
            out: List[Tuple[int, nn.Module]] = []
            for idx in my_indices:
                orig_layer = layers[idx].to(dev)
                if model_cfg is not None and not hasattr(orig_layer, "config"):
                    orig_layer.config = model_cfg
                torch.manual_seed(seed + idx)
                torch.cuda.manual_seed_all(seed + idx)
                compressed = compress(orig_layer, method=method, **kwargs)
                torch.cuda.synchronize(dev)
                out.append((idx, compressed))
            return out

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_gpus) as executor:
            worker_results = list(executor.map(_worker, range(num_gpus)))

        for wr in worker_results:
            for idx, comp_layer in wr:
                layers[idx] = comp_layer.to(target_dev)

        self.context.barrier()
        return model


__all__ = ["LayerParallelExecutor"]
