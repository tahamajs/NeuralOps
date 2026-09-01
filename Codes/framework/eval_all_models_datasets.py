"""Lightweight public injection entry point used by sweep runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn

from framework.core.attention_utils import inject_attention
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY


def apply_module_to_model(model: nn.Module, module_name: str, module_type: str,
                          rank: int = 16, device: torch.device = torch.device("cpu")) -> Tuple[nn.Module, float]:
    original = sum(p.numel() for p in model.parameters())
    if module_name in ATTENTION_REGISTRY:
        module_type = "attention"
    elif module_name in COMPRESSOR_REGISTRY:
        module_type = "compressor"
    if module_type == "attention":
        inject_attention(model, module_name, rank=rank, copy_weights=True)
    elif module_type == "compressor":
        compressor = COMPRESSOR_REGISTRY.resolve(module_name)
        instance = None
        for args, kwargs in (((model,), {"rank": rank}), ((model,), {}), ((), {"rank": rank}), ((), {})):
            try:
                instance = compressor(*args, **kwargs)
                break
            except (TypeError, ValueError):
                continue
        if instance is None:
            raise RuntimeError(f"Could not instantiate compressor {module_name}")
        if isinstance(instance, nn.Module) and not any(callable(getattr(instance, n, None)) for n in ("compress", "prune")):
            model = instance
        else:
            result = None
            # Prioritize dedicated compression methods before generic 'apply' to avoid invoking nn.Module.apply(fn)
            for method_name in ("compress", "prune", "apply"):
                method = getattr(instance, method_name, None)
                if callable(method):
                    # Check if 'apply' is just torch.nn.Module.apply (which expects a fn, not the model)
                    if method_name == "apply" and isinstance(instance, nn.Module) and method.__func__ is nn.Module.apply:
                        continue
                    for call in (lambda: method(model), lambda: method(), lambda: method(calibration_data=None)):
                        try:
                            result = call()
                            break
                        except (TypeError, ValueError, RuntimeError):
                            continue
                    if isinstance(result, tuple):
                        result = result[0]
                    if isinstance(result, nn.Module):
                        model = result
                        break
    else:
        raise RuntimeError(f"Unknown module type: {module_type}")
    model.to(device)
    current = sum(p.numel() for p in model.parameters())
    return model, original / max(1, current)


def build_model_instance(
    family: str,
    num_labels: int = 2,
    hidden_size: int = 64,
    num_heads: int = 4,
    num_layers: int = 1,
    device: torch.device = torch.device("cpu"),
) -> Tuple[nn.Module, Any]:
    """Build a tiny model instance of the given family for testing/eval."""
    family = family.lower()
    if family == "bert":
        from transformers import BertConfig, BertForSequenceClassification
        cfg = BertConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_size * 4,
            num_labels=num_labels,
            vocab_size=30522,
        )
        model = BertForSequenceClassification(cfg)
    elif family == "roberta":
        from transformers import RobertaConfig, RobertaForSequenceClassification
        cfg = RobertaConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            intermediate_size=hidden_size * 4,
            num_labels=num_labels,
            vocab_size=50265,
        )
        model = RobertaForSequenceClassification(cfg)
    elif family == "distilbert":
        from transformers import DistilBertConfig, DistilBertForSequenceClassification
        cfg = DistilBertConfig(
            dim=hidden_size,
            n_heads=num_heads,
            n_layers=num_layers,
            hidden_dim=hidden_size * 4,
            num_labels=num_labels,
            vocab_size=30522,
        )
        model = DistilBertForSequenceClassification(cfg)
    elif family == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel
        cfg = GPT2Config(
            n_embd=hidden_size,
            n_head=num_heads,
            n_layer=num_layers,
            n_inner=hidden_size * 4,
            vocab_size=50257,
        )
        model = GPT2LMHeadModel(cfg)
    elif family == "t5":
        from transformers import T5Config, T5EncoderModel
        cfg = T5Config(
            d_model=hidden_size,
            num_heads=num_heads,
            num_layers=num_layers,
            d_kv=hidden_size // num_heads,
            d_ff=hidden_size * 4,
            vocab_size=32128,
        )
        model = T5EncoderModel(cfg)
    elif family in {"qwen", "qwen2"}:
        from transformers import Qwen2Config, Qwen2ForSequenceClassification
        cfg = Qwen2Config(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=max(1, num_heads // 2),
            num_labels=num_labels,
            vocab_size=1000,
            max_position_embeddings=512,
            pad_token_id=0,
        )
        model = Qwen2ForSequenceClassification(cfg)
    else:
        raise ValueError(f"Unknown model family: {family}")

    model.to(device)
    return model, cfg


def _generate_synthetic_batch(
    dataset_name: str = "sst2",
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 30522,
    num_labels: int = 2,
    device: Any = "cpu",
) -> dict:
    """Generate synthetic batch dict for GLUE and other text classification tasks."""
    dev = torch.device(device)
    b = {
        "input_ids": torch.randint(100, max(101, vocab_size - 100), (batch_size, seq_len), device=dev),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=dev),
        "token_type_ids": torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev),
    }
    if num_labels == 1:
        b["labels"] = torch.randn(batch_size, dtype=torch.float32, device=dev)
    else:
        b["labels"] = torch.randint(0, num_labels, (batch_size,), dtype=torch.long, device=dev)
    return b


def get_dataset_batch(
    family: str,
    batch_size: int = 4,
    seq_len: int = 32,
    device: Any = "cpu",
) -> dict:
    """Generate a synthetic test batch suitable for the given model family."""
    dev = torch.device(device)
    family = family.lower()
    vocab_size = 50265 if family == "roberta" else (50257 if family == "gpt2" else 30522)
    b = {
        "input_ids": torch.randint(100, vocab_size - 100, (batch_size, seq_len), device=dev),
        "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long, device=dev),
    }
    if family not in {"t5", "gpt2"}:
        b["token_type_ids"] = torch.zeros(batch_size, seq_len, dtype=torch.long, device=dev)
    if family == "gpt2":
        b["labels"] = torch.randint(100, vocab_size - 100, (batch_size, seq_len), device=dev)
    elif family != "t5":
        b["labels"] = torch.zeros(batch_size, dtype=torch.long, device=dev)
    return b


def _filter_batch(batch: dict, model: nn.Module) -> dict:
    """Filter batch dictionary keys to only those accepted by model forward pass."""
    import inspect
    forward_params = inspect.signature(model.forward).parameters
    return {k: v for k, v in batch.items() if k in forward_params}


def _extract_logits_and_loss(output: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Extract logits and loss tensor from model output."""
    loss = getattr(output, "loss", None)
    logits = getattr(output, "logits", None)
    if logits is None:
        if isinstance(output, tuple):
            logits = output[0]
            loss = output[1] if len(output) > 1 and isinstance(output[1], torch.Tensor) and output[1].numel() == 1 else None
        elif isinstance(output, torch.Tensor):
            logits = output
        elif hasattr(output, "last_hidden_state"):
            logits = output.last_hidden_state
    return logits, loss


def probe_optimal_batch_size(
    model: nn.Module,
    family: str,
    device: torch.device = torch.device("cpu"),
    max_batch: int = 64,
) -> int:
    """Probe for memory-safe batch size."""
    return min(16, max_batch)

