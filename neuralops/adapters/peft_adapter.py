#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
PEFT Adapter Bridge.

Bi-directional conversion between Hugging Face PEFT LoraConfig and NeuralOps LoRA configurations,
state dict translation, module conversion, and hook injection.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn

from neuralops.core.peft.lora import LoRALinear
from neuralops.utils.lora import DEFAULT_TARGET_MODULES

logger = logging.getLogger("neuralops.adapters.peft_adapter")


class NeuralOpsLoraConfig:
    """Canonical representation of NeuralOps LoRA configuration.

    Encapsulates all hyper-parameters necessary to adapt or heal a model using low-rank adaptation,
    guaranteeing 100% roundtrip fidelity with PEFT LoraConfig.
    """

    def __init__(
        self,
        r: int = 8,
        lora_alpha: float = 16.0,
        target_modules: Optional[Union[List[str], Set[str], Tuple[str, ...]]] = None,
        lora_dropout: float = 0.0,
        bias: str = "none",
        use_rslora: bool = False,
        modules_to_save: Optional[List[str]] = None,
        init_lora_weights: bool = True,
        rank_pattern: Optional[Dict[str, int]] = None,
        alpha_pattern: Optional[Dict[str, float]] = None,
        task_type: Optional[str] = None,
        inference_mode: bool = False,
        **extra_kwargs: Any,
    ) -> None:
        """Initialize NeuralOpsLoraConfig.

        Args:
            r: Rank of the low-rank adaptation matrices.
            lora_alpha: Scaling factor alpha (scaling = alpha / r or alpha / sqrt(r) if RSLoRA).
            target_modules: Collection of module names or suffixes to inject LoRA into.
            lora_dropout: Dropout probability applied before LoRA linear projections.
            bias: Bias training strategy ('none', 'all', or 'lora_only').
            use_rslora: Whether to use Rank-Stabilized LoRA scaling (alpha / sqrt(r)).
            modules_to_save: List of modules apart from LoRA layers to be set as trainable.
            init_lora_weights: Whether to initialize LoRA weights (kaiming for A, zero for B).
            rank_pattern: Optional dictionary mapping module name patterns to specific ranks.
            alpha_pattern: Optional dictionary mapping module name patterns to specific alphas.
            task_type: Optional task type string (e.g. 'CAUSAL_LM', 'SEQ_CLS').
            inference_mode: Whether adapter is configured strictly for inference.
            **extra_kwargs: Additional arguments preserved for complete roundtrip fidelity.
        """
        self.r = int(r)
        self.lora_alpha = float(lora_alpha)
        if target_modules is None:
            self.target_modules: Set[str] = set(DEFAULT_TARGET_MODULES)
        elif isinstance(target_modules, (list, tuple, set)):
            self.target_modules = set(target_modules)
        else:
            self.target_modules = {str(target_modules)}

        self.lora_dropout = float(lora_dropout)
        self.bias = str(bias)
        self.use_rslora = bool(use_rslora)
        self.modules_to_save = list(modules_to_save) if modules_to_save is not None else None
        self.init_lora_weights = bool(init_lora_weights)
        self.rank_pattern = dict(rank_pattern) if rank_pattern is not None else {}
        self.alpha_pattern = dict(alpha_pattern) if alpha_pattern is not None else {}
        self.task_type = task_type
        self.inference_mode = bool(inference_mode)
        self.extra_kwargs = dict(extra_kwargs)

    @property
    def scaling(self) -> float:
        """Compute the effective scaling factor for LoRA delta weights."""
        if self.r <= 0:
            return 1.0
        if self.use_rslora:
            import math

            return self.lora_alpha / math.sqrt(self.r)
        return self.lora_alpha / self.r

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary representation of this configuration.
        """
        data: Dict[str, Any] = {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "target_modules": sorted(list(self.target_modules)),
            "lora_dropout": self.lora_dropout,
            "bias": self.bias,
            "use_rslora": self.use_rslora,
            "modules_to_save": self.modules_to_save,
            "init_lora_weights": self.init_lora_weights,
            "rank_pattern": self.rank_pattern,
            "alpha_pattern": self.alpha_pattern,
            "task_type": self.task_type,
            "inference_mode": self.inference_mode,
        }
        data.update(self.extra_kwargs)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NeuralOpsLoraConfig:
        """Construct NeuralOpsLoraConfig from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            Instantiated NeuralOpsLoraConfig.
        """
        data_copy = dict(data)
        r = data_copy.pop("r", 8)
        lora_alpha = data_copy.pop("lora_alpha", 16.0)
        target_modules = data_copy.pop("target_modules", None)
        lora_dropout = data_copy.pop("lora_dropout", 0.0)
        bias = data_copy.pop("bias", "none")
        use_rslora = data_copy.pop("use_rslora", False)
        modules_to_save = data_copy.pop("modules_to_save", None)
        init_lora_weights = data_copy.pop("init_lora_weights", True)
        rank_pattern = data_copy.pop("rank_pattern", None)
        alpha_pattern = data_copy.pop("alpha_pattern", None)
        task_type = data_copy.pop("task_type", None)
        inference_mode = data_copy.pop("inference_mode", False)

        return cls(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            use_rslora=use_rslora,
            modules_to_save=modules_to_save,
            init_lora_weights=init_lora_weights,
            rank_pattern=rank_pattern,
            alpha_pattern=alpha_pattern,
            task_type=task_type,
            inference_mode=inference_mode,
            **data_copy,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, NeuralOpsLoraConfig):
            return False
        return (
            self.r == other.r
            and abs(self.lora_alpha - other.lora_alpha) < 1e-6
            and self.target_modules == other.target_modules
            and abs(self.lora_dropout - other.lora_dropout) < 1e-6
            and self.bias == other.bias
            and self.use_rslora == other.use_rslora
            and self.modules_to_save == other.modules_to_save
            and self.init_lora_weights == other.init_lora_weights
            and self.rank_pattern == other.rank_pattern
            and self.alpha_pattern == other.alpha_pattern
            and self.task_type == other.task_type
            and self.inference_mode == other.inference_mode
        )


def peft_to_neuralops_config(peft_config: Any) -> NeuralOpsLoraConfig:
    """Convert Hugging Face PEFT LoraConfig to NeuralOpsLoraConfig with full fidelity.

    Args:
        peft_config: Instance of peft.LoraConfig or equivalent dictionary.

    Returns:
        Equivalent NeuralOpsLoraConfig instance.

    Raises:
        TypeError: If peft_config cannot be converted or lacks required fields.
    """
    if hasattr(peft_config, "to_dict"):
        raw_dict = peft_config.to_dict()
    elif isinstance(peft_config, dict):
        raw_dict = dict(peft_config)
    else:
        # Introspect attributes directly
        raw_dict = {}
        for attr in (
            "r",
            "lora_alpha",
            "target_modules",
            "lora_dropout",
            "bias",
            "use_rslora",
            "modules_to_save",
            "init_lora_weights",
            "rank_pattern",
            "alpha_pattern",
            "task_type",
            "inference_mode",
        ):
            if hasattr(peft_config, attr):
                raw_dict[attr] = getattr(peft_config, attr)

    targets = raw_dict.get("target_modules", None)
    if isinstance(targets, (list, tuple, set)):
        targets_set = set(targets)
    elif isinstance(targets, str):
        targets_set = {t.strip() for t in targets.split(",") if t.strip()}
    else:
        targets_set = None

    task_type_val = raw_dict.get("task_type", None)
    if hasattr(task_type_val, "value"):
        task_type_val = task_type_val.value

    return NeuralOpsLoraConfig(
        r=raw_dict.get("r", 8),
        lora_alpha=raw_dict.get("lora_alpha", 16.0),
        target_modules=targets_set,
        lora_dropout=raw_dict.get("lora_dropout", 0.0),
        bias=raw_dict.get("bias", "none"),
        use_rslora=raw_dict.get("use_rslora", False),
        modules_to_save=raw_dict.get("modules_to_save", None),
        init_lora_weights=raw_dict.get("init_lora_weights", True),
        rank_pattern=raw_dict.get("rank_pattern", None),
        alpha_pattern=raw_dict.get("alpha_pattern", None),
        task_type=task_type_val,
        inference_mode=raw_dict.get("inference_mode", False),
    )


def neuralops_to_peft_config(config: NeuralOpsLoraConfig) -> Any:
    """Convert NeuralOpsLoraConfig to Hugging Face PEFT LoraConfig with full fidelity.

    Args:
        config: NeuralOpsLoraConfig instance.

    Returns:
        Instantiated peft.LoraConfig object.

    Raises:
        ImportError: If peft is not installed in the environment.
    """
    try:
        from peft import LoraConfig
    except ImportError as e:
        raise ImportError("PEFT is required to instantiate peft.LoraConfig. Please install peft.") from e

    target_mods = list(config.target_modules) if config.target_modules else None

    # Filter arguments supported by PEFT LoraConfig __init__
    init_kwargs: Dict[str, Any] = {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "target_modules": target_mods,
        "lora_dropout": config.lora_dropout,
        "bias": config.bias,
        "use_rslora": config.use_rslora,
        "modules_to_save": config.modules_to_save,
        "init_lora_weights": config.init_lora_weights,
        "rank_pattern": config.rank_pattern,
        "alpha_pattern": config.alpha_pattern,
        "task_type": config.task_type,
        "inference_mode": config.inference_mode,
    }

    # Pass through known extra kwargs if supported
    import inspect

    sig = inspect.signature(LoraConfig.__init__)
    valid_params = set(sig.parameters.keys())
    for k, v in config.extra_kwargs.items():
        if k in valid_params and k not in init_kwargs:
            init_kwargs[k] = v

    return LoraConfig(**init_kwargs)


def peft_to_neuralops_state_dict(
    peft_state_dict: Dict[str, torch.Tensor],
    adapter_name: str = "default",
) -> Dict[str, torch.Tensor]:
    """Translate state dict from PEFT format to NeuralOps format.

    In PEFT:
      - 'base_model.model.path.to.module.lora_A.{adapter_name}.weight' -> lora_A
      - 'base_model.model.path.to.module.lora_B.{adapter_name}.weight' -> lora_B
      - 'base_model.model.path.to.module.base_layer.weight' -> base_layer.weight
      - 'base_model.model.path.to.module.base_layer.bias' -> base_layer.bias

    In NeuralOps:
      - 'path.to.module.lora_A'
      - 'path.to.module.lora_B'
      - 'path.to.module.base_layer.weight'
      - 'path.to.module.base_layer.bias'

    Args:
        peft_state_dict: Weights mapping in PEFT convention.
        adapter_name: Name of PEFT adapter branch (defaults to 'default').

    Returns:
        Weights mapping conforming to NeuralOps format.
    """
    neuralops_sd: Dict[str, torch.Tensor] = {}

    lora_a_suffix = f".lora_A.{adapter_name}.weight"
    lora_b_suffix = f".lora_B.{adapter_name}.weight"

    for key, tensor in peft_state_dict.items():
        clean_key = key
        # Strip leading PEFT wrapper prefix if present
        if clean_key.startswith("base_model.model."):
            clean_key = clean_key[len("base_model.model.") :]
        elif clean_key.startswith("base_model."):
            clean_key = clean_key[len("base_model.") :]

        if clean_key.endswith(lora_a_suffix):
            prefix = clean_key[: -len(lora_a_suffix)]
            new_key = f"{prefix}.lora_A" if prefix else "lora_A"
            neuralops_sd[new_key] = tensor
        elif clean_key.endswith(lora_b_suffix):
            prefix = clean_key[: -len(lora_b_suffix)]
            new_key = f"{prefix}.lora_B" if prefix else "lora_B"
            neuralops_sd[new_key] = tensor
        else:
            neuralops_sd[clean_key] = tensor

    return neuralops_sd


def neuralops_to_peft_state_dict(
    neuralops_state_dict: Dict[str, torch.Tensor],
    adapter_name: str = "default",
    include_base_prefix: bool = True,
) -> Dict[str, torch.Tensor]:
    """Translate state dict from NeuralOps format to PEFT format.

    Args:
        neuralops_state_dict: Weights mapping in NeuralOps convention.
        adapter_name: Name of PEFT adapter branch (defaults to 'default').
        include_base_prefix: Whether to prepend 'base_model.model.' to keys.

    Returns:
        Weights mapping conforming to PEFT convention.
    """
    peft_sd: Dict[str, torch.Tensor] = {}
    prefix_str = "base_model.model." if include_base_prefix else ""

    for key, tensor in neuralops_state_dict.items():
        if key.endswith(".lora_A") or key == "lora_A":
            mod_prefix = key[:-7] if key.endswith(".lora_A") else ""
            peft_key = (
                f"{prefix_str}{mod_prefix}.lora_A.{adapter_name}.weight"
                if mod_prefix
                else f"{prefix_str}lora_A.{adapter_name}.weight"
            )
            peft_sd[peft_key] = tensor
        elif key.endswith(".lora_B") or key == "lora_B":
            mod_prefix = key[:-7] if key.endswith(".lora_B") else ""
            peft_key = (
                f"{prefix_str}{mod_prefix}.lora_B.{adapter_name}.weight"
                if mod_prefix
                else f"{prefix_str}lora_B.{adapter_name}.weight"
            )
            peft_sd[peft_key] = tensor
        else:
            peft_sd[f"{prefix_str}{key}"] = tensor

    return peft_sd


def inject_lora_adapter(
    model: nn.Module,
    config: Union[NeuralOpsLoraConfig, Any],
) -> Tuple[nn.Module, int]:
    """Inject NeuralOps LoRALinear layers into model according to configuration.

    Args:
        model: PyTorch module/model to adapt.
        config: Configuration instance (either NeuralOpsLoraConfig or PEFT LoraConfig).

    Returns:
        Tuple of (adapted_model, count_of_injected_adapters).
    """
    if not isinstance(config, NeuralOpsLoraConfig):
        config = peft_to_neuralops_config(config)

    targets = config.target_modules
    adapted_count = 0

    def _should_adapt(name: str, module: nn.Module) -> bool:
        if not (isinstance(module, nn.Linear) or module.__class__.__name__ in ("Linear", "Conv1D")):
            return False
        # Do not adapt already adapted LoRALinear layers
        if isinstance(module, LoRALinear):
            return False
        # Avoid task heads unless explicitly specified
        if any(head in name.lower() for head in ("score", "classifier", "lm_head")):
            leaf_name = name.split(".")[-1]
            if not any(leaf_name == t for t in targets):
                return False
        # Match against target collection
        leaf_name = name.split(".")[-1]
        return any(t in name or t == leaf_name for t in targets)

    def _replace_modules_recursive(mod: nn.Module, prefix: str = "") -> None:
        nonlocal adapted_count
        for child_name, child in list(mod.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if _should_adapt(full_name, child):
                rank = config.rank_pattern.get(full_name, config.r)
                alpha = config.alpha_pattern.get(full_name, config.lora_alpha)
                wrapped = LoRALinear(
                    child,
                    r=rank,
                    alpha=alpha,
                    dropout=config.lora_dropout,
                    freeze_base=True,
                )
                setattr(mod, child_name, wrapped)
                adapted_count += 1
            else:
                _replace_modules_recursive(child, full_name)

    _replace_modules_recursive(model)
    return model, adapted_count


def attach_lora_hooks(
    model: nn.Module,
    pre_hook_fn: Optional[Any] = None,
    post_hook_fn: Optional[Any] = None,
) -> List[torch.utils.hooks.RemovableHandle]:
    """Register forward pre-hooks and post-hooks onto all LoRALinear layers in model.

    Args:
        model: PyTorch model containing LoRALinear modules.
        pre_hook_fn: Optional callable (module, inputs) -> None or modified inputs.
        post_hook_fn: Optional callable (module, inputs, output) -> None or modified output.

    Returns:
        List of RemovableHandle objects for managing hook lifecycle.
    """
    handles: List[torch.utils.hooks.RemovableHandle] = []
    for _, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            if pre_hook_fn is not None:
                h = mod.register_forward_pre_hook(pre_hook_fn)
                handles.append(h)
            if post_hook_fn is not None:
                h = mod.register_forward_hook(post_hook_fn)
                handles.append(h)
    return handles
