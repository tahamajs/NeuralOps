#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Universal Model Adapters for NeuralOps Benchmarks.

Provides decoupled model wrapper classes (BaseModelAdapter, HuggingFaceModelAdapter,
SurgerizedModelAdapter, UniversalModelAdapter) to abstract forward evaluation,
text generation, logit extraction, and hidden state inspection across arbitrary
foundation models, surgerized checkpoints, and in-memory architectures.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

try:
    from neuralops.surgery.hf_cache_bridge import (
        NeuralOpsHybridHFCache,
        PolymorphicGenerationCacheBridge,
        wrap_model_for_generation,
    )
    _HAS_CACHE_BRIDGE = True
except ImportError:
    _HAS_CACHE_BRIDGE = False
    NeuralOpsHybridHFCache = None  # type: ignore[assignment,misc]
    PolymorphicGenerationCacheBridge = None  # type: ignore[assignment,misc]
    wrap_model_for_generation = None  # type: ignore[assignment]

try:
    from transformers.cache_utils import DynamicCache
    _HAS_DYNAMIC_CACHE = True
except ImportError:
    _HAS_DYNAMIC_CACHE = False
    DynamicCache = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class BaseModelAdapter(ABC):
    """Abstract base class defining the universal model interface for benchmarking."""

    def __init__(self, model: nn.Module, model_name: str = "custom_model") -> None:
        self.model = model
        self.model_name = model_name
        self._device = self._detect_device()

    def _detect_device(self) -> torch.device:
        """Infer active device from model parameters or buffers."""
        try:
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Return the device where the underlying model resides."""
        return self._device

    def to(self, device: Union[str, torch.device]) -> "BaseModelAdapter":
        """Move underlying model to specified device."""
        self._device = torch.device(device)
        self.model.to(self._device)
        return self

    def eval(self) -> "BaseModelAdapter":
        """Set underlying model to evaluation mode."""
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> "BaseModelAdapter":
        """Set underlying model to training mode."""
        self.model.train(mode)
        return self

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Standard forward pass through the model."""

    @abstractmethod
    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract next-token / classification logits tensor of shape (batch, seq_len, vocab_size) or (batch, num_labels)."""

    @abstractmethod
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_indices: Optional[Union[int, Sequence[int]]] = None,
        layer_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Extract intermediate hidden states across layers or at specified layer indices."""

    def extract_pooled_output(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pooling_mode: str = "mean",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract sequence-level pooled representation for encoders / representation analysis.

        Supported pooling modes:
        - 'mean': Mean pooling over tokens respecting attention_mask.
        - 'cls': First token ([CLS]) embedding representation.
        - 'max': Max pooling over tokens.
        - 'last': Last token representation.
        """
        states = self.get_hidden_states(input_ids, attention_mask=attention_mask, **kwargs)
        if isinstance(states, (tuple, list)):
            last_hidden = states[-1]
        else:
            last_hidden = states

        if pooling_mode == "cls":
            return last_hidden[:, 0, :]
        elif pooling_mode == "last":
            if attention_mask is not None:
                seq_lengths = attention_mask.sum(dim=1).long() - 1
                batch_size = last_hidden.size(0)
                return last_hidden[torch.arange(batch_size, device=last_hidden.device), seq_lengths]
            return last_hidden[:, -1, :]
        elif pooling_mode == "max":
            if attention_mask is not None:
                expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
                last_hidden = last_hidden.masked_fill(expanded_mask == 0, -1e9)
            return torch.max(last_hidden, dim=1)[0]
        else:  # 'mean'
            if attention_mask is not None:
                expanded_mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
                sum_embeddings = torch.sum(last_hidden * expanded_mask, dim=1)
                sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            return torch.mean(last_hidden, dim=1)

    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressively generate token sequences."""

    def count_parameters(self) -> Dict[str, int]:
        """Inspect total and trainable parameter counts."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total_params": total, "trainable_params": trainable}

    def get_layer_count(self) -> int:
        """Estimate number of transformer layers in model."""
        if hasattr(self.model, "config") and hasattr(self.model.config, "num_hidden_layers"):
            return int(self.model.config.num_hidden_layers)
        if hasattr(self.model, "layers") and isinstance(self.model.layers, (nn.ModuleList, list)):
            return len(self.model.layers)
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return len(self.model.transformer.h)
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            return len(self.model.encoder.layer)
        # Fallback inspection by scanning child modules named 'layer' or 'block'
        layer_submodules = [
            m for name, m in self.model.named_modules()
            if any(k in name.lower() for k in [".layer.", ".layers.", ".block.", ".h."]) and len(list(m.children())) == 0
        ]
        return max(1, len(layer_submodules))

    def get_model_metadata(self) -> Dict[str, Any]:
        """Return structural inspection metadata."""
        params = self.count_parameters()
        return {
            "model_name": self.model_name,
            "adapter_type": self.__class__.__name__,
            "total_params": params["total_params"],
            "trainable_params": params["trainable_params"],
            "num_layers": self.get_layer_count(),
            "device": str(self.device),
        }


class HuggingFaceModelAdapter(BaseModelAdapter):
    """Adapter wrapping HuggingFace PreTrainedModel instances (CausalLM or SequenceClassification)."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "hf_model",
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__(model=model, model_name=model_name)
        self.tokenizer = tokenizer

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def get_logits(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Extract prediction logits from HuggingFace model output."""
        kwargs_fwd = dict(kwargs)
        if attention_mask is not None:
            kwargs_fwd["attention_mask"] = attention_mask

        outputs = self.model(input_ids, **kwargs_fwd)
        if hasattr(outputs, "logits"):
            return outputs.logits
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        if isinstance(outputs, torch.Tensor):
            return outputs
        raise AttributeError(f"Unable to extract logits from output type: {type(outputs)}")

    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_indices: Optional[Union[int, Sequence[int]]] = None,
        layer_idx: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Extract hidden states via output_hidden_states=True."""
        kwargs_fwd = dict(kwargs)
        kwargs_fwd["output_hidden_states"] = True
        if attention_mask is not None:
            kwargs_fwd["attention_mask"] = attention_mask

        outputs = self.model(input_ids, **kwargs_fwd)
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            raise RuntimeError(
                f"Model {self.model_name} failed to return hidden_states despite output_hidden_states=True."
            )

        states = outputs.hidden_states
        if layer_idx is not None:
            return states[layer_idx]
        if layer_indices is not None:
            if isinstance(layer_indices, int):
                return states[layer_indices]
            return tuple(states[i] for i in layer_indices)
        return states

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Autoregressive generation via HF .generate() or PolymorphicGenerationCacheBridge.

        Interfaces with DynamicCache / PolymorphicGenerationCacheBridge for O(1) recurrent step latency.
        """
        # If model supports HF generate, wrap with PolymorphicGenerationCacheBridge if available
        if hasattr(self.model, "generate"):
            try:
                wrapped_model = self.model
                if _HAS_CACHE_BRIDGE and wrap_model_for_generation is not None:
                    wrapped_model = wrap_model_for_generation(self.model)

                gen_kwargs: Dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "use_cache": use_cache,
                    **kwargs,
                }
                if do_sample:
                    gen_kwargs["temperature"] = temperature
                    if top_k is not None:
                        gen_kwargs["top_k"] = top_k
                    if top_p is not None:
                        gen_kwargs["top_p"] = top_p
                if eos_token_id is not None:
                    gen_kwargs["eos_token_id"] = eos_token_id

                return wrapped_model.generate(input_ids, **gen_kwargs)
            except Exception as e:
                logger.warning("HF native generate failed (%s); falling back to manual generation loop", e)

        # Autoregressive loop fallback with DynamicCache / PolymorphicGenerationCacheBridge support
        curr_ids = input_ids.clone()
        past_key_values: Optional[Any] = None
        if use_cache:
            if _HAS_CACHE_BRIDGE and PolymorphicGenerationCacheBridge is not None:
                past_key_values = PolymorphicGenerationCacheBridge()
            elif _HAS_DYNAMIC_CACHE and DynamicCache is not None:
                past_key_values = DynamicCache()

        for step in range(max_new_tokens):
            with torch.no_grad():
                step_input = curr_ids if past_key_values is None else curr_ids[:, -1:]
                step_kwargs: Dict[str, Any] = dict(kwargs)
                if past_key_values is not None:
                    step_kwargs["past_key_values"] = past_key_values
                    step_kwargs["use_cache"] = True

                outputs = self.model(step_input, **step_kwargs)

                # Extract logits and update cache
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                elif isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                elif isinstance(outputs, torch.Tensor):
                    logits = outputs
                else:
                    raise AttributeError(f"Unable to extract logits in generate loop: {type(outputs)}")

                if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
                    past_key_values = outputs.past_key_values
                elif isinstance(outputs, (tuple, list)) and len(outputs) > 1 and outputs[1] is not None:
                    past_key_values = outputs[1]

                next_token_logits = logits[:, -1, :]
                if do_sample and temperature > 0:
                    probs = torch.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                curr_ids = torch.cat([curr_ids, next_token], dim=-1)
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        return curr_ids


class SurgerizedModelAdapter(HuggingFaceModelAdapter):
    """Adapter wrapping models modified via NeuralOps surgery with lineage and calibration metadata."""

    def __init__(
        self,
        model: nn.Module,
        model_name: str = "surgerized_model",
        surgery_manifest: Optional[Dict[str, Any]] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        super().__init__(model=model, model_name=model_name, tokenizer=tokenizer)
        self.surgery_manifest = surgery_manifest or {}
        self.transplanted_attention = str(
            self.surgery_manifest.get("transplanted_attention")
            or self.surgery_manifest.get("attention_key")
            or ""
        )
        self.compressor_key = str(
            self.surgery_manifest.get("compressor_key")
            or self.surgery_manifest.get("compression_method")
            or ""
        )
        self.calibrated = bool(self.surgery_manifest.get("calibrated", False))
        self.ridge_damping = str(
            self.surgery_manifest.get("ridge_damping")
            or self.surgery_manifest.get("damping")
            or "1e-4"
        )
        self.pre_train_healed = str(
            self.surgery_manifest.get("pre_train_healed")
            or self.surgery_manifest.get("healed")
            or ("true" if self.calibrated else "false")
        )

    def get_surgery_lineage(self) -> Dict[str, Any]:
        """Return provenance and surgical lineage metadata.

        Preserves lineage tracking attributes:
        transplanted_attention, compressor_key, ridge_damping, calibrated, pre_train_healed.
        """
        return {
            "is_transplanted": bool(self.transplanted_attention != ""),
            "transplanted_attention": self.transplanted_attention,
            "compressor_key": self.compressor_key,
            "ridge_damping": self.ridge_damping,
            "calibrated": "true" if self.calibrated else "false",
            "pre_train_healed": self.pre_train_healed,
            "calibration_method": self.surgery_manifest.get("calibration_method", "none"),
            "base_model": self.surgery_manifest.get("base_model", self.model_name),
            "surgery_manifest": self.surgery_manifest,
        }

    @classmethod
    def from_checkpoint_dir(
        cls,
        checkpoint_dir: Union[str, Path],
        model_name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> "SurgerizedModelAdapter":
        """Instantiate SurgerizedModelAdapter directly from a checkpoint directory.

        Parses and validates 'neuralops_surgery_manifest.json' and locates accompanying weights.
        Fails closed with FileNotFoundError or ValueError on missing or corrupt manifests.
        """
        ckpt_path = Path(checkpoint_dir)
        if not ckpt_path.exists() or not ckpt_path.is_dir():
            raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_path}")

        manifest_file = ckpt_path / "neuralops_surgery_manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(
                f"Missing required surgery manifest 'neuralops_surgery_manifest.json' in {ckpt_path}"
            )

        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            raise ValueError(f"Corrupted or invalid JSON in surgery manifest {manifest_file}: {e}") from e

        if not isinstance(manifest, dict):
            raise ValueError(f"Invalid schema: manifest in {manifest_file} must be a JSON object")

        # Load weights using transformers AutoModelForCausalLM or AutoModel
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            load_kwargs: Dict[str, Any] = dict(kwargs)
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), **load_kwargs)
            if tokenizer is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
                except Exception:
                    pass
        except Exception as e:
            raise RuntimeError(f"Failed to load weights from checkpoint dir {ckpt_path}: {e}") from e

        resolved_name = model_name or manifest.get("model_tag") or ckpt_path.name
        adapter = cls(
            model=model,
            model_name=resolved_name,
            surgery_manifest=manifest,
            tokenizer=tokenizer,
        )
        if device is not None:
            adapter.to(device)
        return adapter

    def get_model_metadata(self) -> Dict[str, Any]:
        meta = super().get_model_metadata()
        meta["surgery_lineage"] = self.get_surgery_lineage()
        return meta


class UniversalModelAdapter:
    """Universal factory and dispatcher converting various model formats into a BaseModelAdapter."""

    @staticmethod
    def from_pretrained(
        model_or_path: Union[str, Path, nn.Module],
        model_name: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        torch_dtype: Optional[torch.dtype] = None,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> BaseModelAdapter:
        """Instantiate an adapter from a path, checkpoint directory, HuggingFace hub id, or nn.Module.

        Detects whether the target is:
        1. An already-wrapped BaseModelAdapter.
        2. A local directory containing 'neuralops_surgery_manifest.json' -> SurgerizedModelAdapter.
        3. A local or hub HuggingFace model -> HuggingFaceModelAdapter.
        4. A raw PyTorch nn.Module -> HuggingFaceModelAdapter / BaseModelAdapter.
        """
        if isinstance(model_or_path, BaseModelAdapter):
            if device is not None:
                model_or_path.to(device)
            return model_or_path

        # Case 1: Local checkpoint directory containing surgery manifest
        if isinstance(model_or_path, (str, Path)) and os.path.isdir(str(model_or_path)):
            ckpt_path = Path(model_or_path)
            manifest_file = ckpt_path / "neuralops_surgery_manifest.json"
            manifest: Dict[str, Any] = {}
            if manifest_file.is_file():
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                except Exception as e:
                    logger.warning("Could not parse surgery manifest %s: %s", manifest_file, e)

            # Attempt loading via AutoModelForCausalLM or AutoModel
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                load_kwargs: Dict[str, Any] = dict(kwargs)
                if torch_dtype is not None:
                    load_kwargs["torch_dtype"] = torch_dtype
                model = AutoModelForCausalLM.from_pretrained(str(ckpt_path), **load_kwargs)
                if tokenizer is None:
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(str(ckpt_path))
                    except Exception:
                        pass
                resolved_name = model_name or manifest.get("model_tag") or ckpt_path.name
                adapter = SurgerizedModelAdapter(
                    model=model,
                    model_name=resolved_name,
                    surgery_manifest=manifest,
                    tokenizer=tokenizer,
                )
                if device is not None:
                    adapter.to(device)
                return adapter
            except Exception as e:
                logger.debug("Loading directory via AutoModelForCausalLM failed (%s); checking generic loader", e)

        # Case 2: HuggingFace model string identifier
        if isinstance(model_or_path, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            load_kwargs = dict(kwargs)
            if torch_dtype is not None:
                load_kwargs["torch_dtype"] = torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_or_path, **load_kwargs)
            if tokenizer is None:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_or_path)
                except Exception:
                    pass
            adapter = HuggingFaceModelAdapter(
                model=model,
                model_name=model_name or model_or_path,
                tokenizer=tokenizer,
            )
            if device is not None:
                adapter.to(device)
            return adapter

        # Case 3: Instantiated nn.Module
        if isinstance(model_or_path, nn.Module):
            resolved_name = model_name or getattr(model_or_path, "name", model_or_path.__class__.__name__)
            surgery_manifest = getattr(model_or_path, "surgery_manifest", None)
            if surgery_manifest is not None or getattr(model_or_path, "is_transplanted", False):
                adapter = SurgerizedModelAdapter(
                    model=model_or_path,
                    model_name=resolved_name,
                    surgery_manifest=surgery_manifest or {},
                    tokenizer=tokenizer,
                )
            else:
                adapter = HuggingFaceModelAdapter(
                    model=model_or_path,
                    model_name=resolved_name,
                    tokenizer=tokenizer,
                )
            if device is not None:
                adapter.to(device)
            return adapter

        raise TypeError(f"Unsupported model input type: {type(model_or_path)}")

    create = from_pretrained


__all__ = [
    "BaseModelAdapter",
    "HuggingFaceModelAdapter",
    "SurgerizedModelAdapter",
    "UniversalModelAdapter",
]
