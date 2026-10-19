#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Attention Layer Replacement Utility

Replaces native model attention mechanisms with linear and factorized attention modules.
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from neuralops.utils.qwen_compat import bypass_qwen_rope, qwen_attention_mask


class UniversalAttentionWrapper(nn.Module):
    """
    Universal forward adapter wrapping any custom attention module.
    Seamlessly adapts across all HuggingFace architectures:
      - DistilBERT: forward(query=x, key=x, value=x, mask=mask, head_mask=head_mask, output_attentions=...) -> tuple (out, weights)
      - BERT / RoBERTa: forward(hidden_states, attention_mask, ...) -> tuple (out, weights)
      - LLaMA / Mistral: forward(hidden_states, attention_mask, position_ids, ...) -> tuple (out, weights, ...)
      - GPT-2: forward(x, layer_past, ...) -> tuple (out, present)
    """

    def __init__(
        self,
        inner_attn: nn.Module,
        layer_name: str = "",
        return_tuple: bool | None = None,
    ) -> None:
        super().__init__()
        self._wrapped_attention = inner_attn
        self.layer_name = layer_name
        self.return_tuple = return_tuple
        self.is_t5: bool = False
        self.has_relative_attention_bias = None
        self.relative_attention_bias = None
        self.relative_attention_num_buckets = None
        self.relative_attention_max_distance = None
        self.is_decoder = None
        self.layer_idx = None
        self.compute_bias = None

    @property
    def inner(self) -> nn.Module:
        return self._wrapped_attention

    @inner.setter
    def inner(self, module: nn.Module) -> None:
        self._wrapped_attention = module

    def forward(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert callable(self._wrapped_attention), f"{type(self).__name__}._wrapped_attention is not callable"
        hidden_states: torch.Tensor | None = None
        if len(args) > 0 and isinstance(args[0], tuple):
            args = (args[0][0],) + args[1:]
        if len(args) > 0 and isinstance(args[0], torch.Tensor):
            hidden_states = args[0]
            args = args[1:]
        elif "hidden_states" in kwargs:
            hidden_states = kwargs.pop("hidden_states")
        elif "query" in kwargs:
            hidden_states = kwargs.pop("query")
        elif "x" in kwargs:
            hidden_states = kwargs.pop("x")

        attention_mask: torch.Tensor | None = None
        if len(args) > 0 and (isinstance(args[0], torch.Tensor) or args[0] is None):
            attention_mask = args[0]
            args = args[1:]
        elif "attention_mask" in kwargs:
            attention_mask = kwargs.pop("attention_mask")
        elif "mask" in kwargs:
            attention_mask = kwargs.pop("mask")
        elif "attn_mask" in kwargs:
            attention_mask = kwargs.pop("attn_mask")

        if "position_embeddings" in kwargs:
            if not getattr(self._wrapped_attention, "supports_rope", False):
                kwargs = bypass_qwen_rope(kwargs, self._wrapped_attention)
                kwargs.pop("qwen_rope_bypassed", None)
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 4:
                attention_mask = qwen_attention_mask(
                    attention_mask,
                    causal=True,
                    expects_bidirectional=bool(
                        getattr(self._wrapped_attention, "expects_bidirectional", False)
                    ),
                )

        key_value_states = kwargs.pop("key_value_states", None)
        position_bias = kwargs.pop("position_bias", None)

        from neuralops.core.t5_compat import is_t5_model

        cfg = getattr(self._wrapped_attention, "config", getattr(self, "config", None))
        is_t5 = (
            getattr(self, "is_t5", False)
            or getattr(self._wrapped_attention, "is_t5", False)
            or is_t5_model(cfg)
            or key_value_states is not None
            or position_bias is not None
            or (getattr(self, "relative_attention_bias", None) is not None)
            or (getattr(self._wrapped_attention, "relative_attention_bias", None) is not None)
        )

        if is_t5:
            out, ret_pos_bias = self._forward_t5(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                **kwargs,
            )
            return out, ret_pos_bias, None

        kwargs.pop("key", None)
        kwargs.pop("value", None)
        head_mask = kwargs.pop("head_mask", None)
        output_attentions = kwargs.pop("output_attentions", False)

        out = None
        call_attempts = [
            lambda: self._wrapped_attention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                output_attentions=output_attentions,
                **kwargs,
            ),
            lambda: self._wrapped_attention(hidden_states, attention_mask=attention_mask, **kwargs),
            lambda: self._wrapped_attention(hidden_states, mask=attention_mask, **kwargs),
            lambda: self._wrapped_attention(hidden_states, **kwargs),
            lambda: self._wrapped_attention(
                query=hidden_states,
                key=hidden_states,
                value=hidden_states,
                mask=attention_mask,
                **kwargs,
            ),
            lambda: self._wrapped_attention(
                hidden_states,
                **{k: v for k, v in kwargs.items() if k in ("position_embeddings", "past_key_value", "past_key_values", "use_cache", "cache_position")}
            ),
            lambda: self._wrapped_attention(hidden_states, attention_mask=attention_mask),
            lambda: self._wrapped_attention(hidden_states),
        ]

        last_err = None
        for attempt in call_attempts:
            try:
                out = attempt()
                break
            except TypeError as e:
                last_err = e
                continue


        if out is None:
            if last_err is not None:
                raise last_err
            raise RuntimeError(
                f"Failed to execute forward pass on {type(self._wrapped_attention).__name__}"
            )

        from neuralops.utils.compat import is_decoder_family
        from neuralops.utils.model_family import detect_family

        cfg = getattr(self._wrapped_attention, "config", getattr(self, "config", None))
        fam = detect_family(cfg) if cfg is not None else "unknown"
        is_decoder = is_decoder_family(fam) or fam in {
            "qwen3_5",
            "qwen",
            "gpt",
            "llama",
            "mistral",
            "ssm",
        }

        incoming_cache = kwargs.get("past_key_value", kwargs.get("past_key_values", None))
        if isinstance(out, tuple):
            hidden = out[0] if len(out) >= 1 else None
            present = out[1] if len(out) >= 2 else incoming_cache
            attn_weights = out[2] if len(out) >= 3 else None
        else:
            hidden = out
            present = incoming_cache
            attn_weights = None

        if self.return_tuple is not None and not self.return_tuple:
            return hidden

        if "linear_attn" in self.layer_name and not output_attentions:
            return hidden

        if fam in ("llama", "mistral", "qwen", "qwen3_5", "qwen2", "qwen2_5") or (
            hasattr(cfg, "model_type")
            and getattr(cfg, "model_type", "").lower() in ("llama", "mistral", "qwen2", "qwen2_5", "qwen")
        ):
            try:
                import transformers
                from packaging import version
                is_legacy_hf_decoder = version.parse(transformers.__version__) < version.parse("4.49.0")
            except (ImportError, AttributeError, ValueError):
                is_legacy_hf_decoder = False

            if is_legacy_hf_decoder:
                if present is None:
                    try:
                        from transformers.cache_utils import DynamicCache
                        present = DynamicCache()
                    except (ImportError, AttributeError, TypeError):
                        present = ()
                return (hidden, attn_weights, present)
            # Transformers 5.x decoder layers unpack exactly two values from
            # self-attention. Cache state is owned by the caller's Cache API.
            return (hidden, attn_weights)

        if output_attentions:
            return (hidden, present, attn_weights)
        return (hidden, present)
    def _forward_t5(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """T5 forward pass executing relative position bias and cross/self attention correctly."""
        if hasattr(self._wrapped_attention, "forward_t5") and callable(
            self._wrapped_attention.forward_t5
        ):
            res = self._wrapped_attention.forward_t5(
                hidden_states,
                attention_mask=attention_mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                **kwargs,
            )
            if isinstance(res, tuple):
                return res[0], (res[1] if len(res) > 1 else position_bias)
            return res, position_bias

        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states
        B, N_q, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H = getattr(self._wrapped_attention, "num_heads", getattr(self, "num_heads", 8))
        d = getattr(self._wrapped_attention, "head_dim", getattr(self, "head_dim", 64))

        if position_bias is None:
            compute_bias_fn = getattr(
                self._wrapped_attention,
                "compute_bias",
                getattr(self, "compute_bias", None),
            )
            has_rel = getattr(
                self._wrapped_attention,
                "has_relative_attention_bias",
                getattr(self, "has_relative_attention_bias", False),
            )
            rel_emb = getattr(
                self._wrapped_attention,
                "relative_attention_bias",
                getattr(self, "relative_attention_bias", None),
            )

            if has_rel and compute_bias_fn is not None and rel_emb is not None:
                position_bias = compute_bias_fn(N_q, N_k, device=hidden_states.device)
            else:
                position_bias = torch.zeros(
                    (1, H, N_q, N_k),
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

            if attention_mask is not None:
                causal_mask = attention_mask[:, :, :, :N_k] if attention_mask.dim() == 4 else None
                if causal_mask is not None:
                    position_bias = position_bias + causal_mask.to(position_bias.dtype)

        q_mod = getattr(
            self._wrapped_attention,
            "q_proj",
            getattr(
                self._wrapped_attention,
                "q",
                getattr(self._wrapped_attention, "query", None),
            ),
        )
        k_mod = getattr(
            self._wrapped_attention,
            "k_proj",
            getattr(
                self._wrapped_attention,
                "k",
                getattr(self._wrapped_attention, "key", None),
            ),
        )
        v_mod = getattr(
            self._wrapped_attention,
            "v_proj",
            getattr(
                self._wrapped_attention,
                "v",
                getattr(self._wrapped_attention, "value", None),
            ),
        )
        o_mod = getattr(
            self._wrapped_attention,
            "out_proj",
            getattr(
                self._wrapped_attention,
                "o",
                getattr(self._wrapped_attention, "dense", None),
            ),
        )

        if q_mod is not None and k_mod is not None and v_mod is not None:
            Q = q_mod(hidden_states).view(B, N_q, H, d).transpose(1, 2)

            k_out = k_mod(kv_input)
            H_kv = k_out.shape[-1] // d if d > 0 else H
            K = k_out.view(B, N_k, H_kv, d).transpose(1, 2)
            if H_kv < H:
                K = K.repeat_interleave(H // H_kv, dim=1)

            v_out = v_mod(kv_input)
            H_v = v_out.shape[-1] // d if d > 0 else H
            V = v_out.view(B, N_k, H_v, d).transpose(1, 2)
            if H_v < H:
                V = V.repeat_interleave(H // H_v, dim=1)

            scores = torch.matmul(Q, K.transpose(-2, -1))
            if position_bias is not None:
                scores = scores + position_bias.to(scores.dtype)

            if attention_mask is not None and attention_mask.dim() == 2:
                min_val = torch.finfo(scores.dtype).min if scores.dtype.is_floating_point else -1e4
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(2) == 0
                scores = scores.masked_fill(extended_mask, min_val)

            import torch.nn.functional as F

            probs = F.softmax(scores.float(), dim=-1).to(scores.dtype)
            probs = torch.nan_to_num(probs, nan=0.0)

            dropout = getattr(self._wrapped_attention, "dropout", None)
            if dropout is not None and callable(dropout):
                probs = dropout(probs)

            context = torch.matmul(probs, V).transpose(1, 2).contiguous().view(B, N_q, -1)
            output = o_mod(context) if o_mod is not None else context
            return output, position_bias

        try:
            res = self._wrapped_attention(
                hidden_states,
                key_value_states=key_value_states,
                encoder_hidden_states=key_value_states,
                **kwargs,
            )
        except TypeError:
            res = self._wrapped_attention(hidden_states, **kwargs)
        if isinstance(res, tuple):
            return res[0], (res[1] if len(res) > 1 and res[1] is not None else position_bias)
        return res, position_bias

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ("_wrapped_attention", "_modules", "_parameters", "_buffers"):
                raise
            return getattr(self._wrapped_attention, name)


def _try_instantiate(
    attn_fn: Any,
    config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
    **kwargs: Any,
) -> nn.Module:
    """
    Instantiate a registered attention function / class using multiple
    call-signature strategies, from most-specific to most-permissive.

    Handles all 4 observed patterns across 126 registry entries:
      1. inject_fn(layer, model_config, comp_config, layer_idx, total_layers, old_attn)
      2. inject_fn(model_config, layer_idx=N, total_layers=N)
      3. AttnClass(config, layer_idx=N, total_layers=N)
      4. AttnClass(config)
    """
    for attr, default in [
        ("dropout", 0.1),
        ("hidden_dropout_prob", 0.1),
        ("attention_probs_dropout_prob", 0.1),
        ("intermediate_size", 3072),
        ("max_position_embeddings", 512),
        ("layer_norm_eps", 1e-12),
        ("is_decoder", False),
        ("add_cross_attention", False),
        ("chunk_size_feed_forward", 0),
        ("position_embedding_type", "absolute"),
    ]:
        if not hasattr(config, attr):
            setattr(config, attr, default)

    _default_ranks = {"q": 64, "k": 64, "v": 64, "o": 64}

    if isinstance(attn_fn, type) and issubclass(attn_fn, nn.Module):
        class_attempts = [
            lambda: attn_fn(config=config, pretrained_attn=old_attn, **kwargs),
            lambda: attn_fn(config=config, **kwargs),
            lambda: attn_fn(config, pretrained_attn=old_attn),
            lambda: attn_fn(config, layer_idx=layer_idx, total_layers=total_layers),
            lambda: attn_fn(config),
            lambda: attn_fn(config.hidden_size, getattr(config, "num_attention_heads", 12)),
            lambda: attn_fn(config, config.hidden_size, getattr(config, "num_attention_heads", 12)),
        ]
        for attempt in class_attempts:
            try:
                result = attempt()
                if isinstance(result, nn.Module):
                    return result
            except (TypeError, ValueError, AttributeError):
                continue

    attempts = [
        lambda: attn_fn(old_attn, config, None, layer_idx, total_layers, old_attn),
        lambda: attn_fn(old_attn, config, None, layer_idx, total_layers),
        lambda: attn_fn(
            old_attn,
            config,
            comp_config=None,
            layer_idx=layer_idx,
            total_layers=total_layers,
        ),
        lambda: attn_fn(old_attn, config),
        lambda: attn_fn(
            config,
            comp_config=None,
            old_attn=old_attn,
            layer_idx=layer_idx,
            total_layers=total_layers,
        ),
        lambda: attn_fn(config, layer_idx=layer_idx, total_layers=total_layers),
        lambda: attn_fn(config, comp_config=None, old_attn=old_attn),
        lambda: attn_fn(
            config, pretrained_attn=old_attn, ranks=_default_ranks, layer_idx=layer_idx
        ),
        lambda: attn_fn(config, pretrained_attn=old_attn, ranks=_default_ranks),
        lambda: attn_fn(config, pretrained_attn=old_attn, layer_idx=layer_idx),
        lambda: attn_fn(config, pretrained_attn=old_attn),
        lambda: attn_fn(config, k_rank=32, pretrained_attn=old_attn),
        lambda: attn_fn(config, None, pretrained_attn=old_attn),
        lambda: attn_fn(config, None),
        lambda: attn_fn(None, None),
        lambda: attn_fn(config, config.hidden_size, getattr(config, "num_attention_heads", 12)),
        lambda: attn_fn(config),
        lambda: attn_fn(config.hidden_size, getattr(config, "num_attention_heads", 12)),
        lambda: attn_fn(old_attn, 32, 32),
        lambda: attn_fn(old_attn, 0.95),
        lambda: attn_fn(old_attn),
        lambda: attn_fn(),
    ]

    last_err = None
    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, nn.Module):
                return result
        except (TypeError, AttributeError, ValueError) as e:
            last_err = e
            continue
        except RuntimeError as e:
            last_err = e
            continue

    raise RuntimeError(
        f"Could not instantiate {getattr(attn_fn, '__name__', str(attn_fn))} "
        f"with any known signature. Last error: {last_err}"
    )


def replace_attention(model, attention_key=None, layer_indices=None, new_attention_cls=None, **kwargs) -> Any:
    """
    Replace attention modules in a model with a registered attention block or class.
    """
    if new_attention_cls is not None:
        attn_fn = new_attention_cls
    elif attention_key is not None:
        if isinstance(attention_key, type) or callable(attention_key):
            attn_fn = attention_key
        else:
            from neuralops.core.registry import ATTENTION_REGISTRY
            attn_fn = ATTENTION_REGISTRY.get(attention_key)
            if attn_fn is None:
                raise ValueError(
                    f"Attention key '{attention_key}' not found in registry. "
                    f"Available: {list(ATTENTION_REGISTRY.keys())}"
                )
    else:
        raise ValueError("Either 'attention_key' or 'new_attention_cls' must be provided.")

    if not hasattr(model, "config"):
        raise AttributeError("Model does not have a 'config' attribute.")

    config = model.config
    total_layers = getattr(config, "num_hidden_layers", 12)

    targets: list[tuple[str, nn.Module, int]] = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if not parts or not parts[-1]:
            continue
        leaf = parts[-1].lower()
        if leaf in [
            "attention",
            "self_attn",
            "self",
            "attn",
            "selfattention",
            "encdecattention",
        ] or isinstance(module, nn.MultiheadAttention):
            if not any(name.startswith(t[0] + ".") for t in targets):
                layer_idx = 0
                for part in parts:
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                if layer_indices is None or layer_idx in layer_indices:
                    targets.append((name, module, layer_idx))

    replaced_count = 0
    failed_count = 0
    for name, old_module, layer_idx in targets:
        try:
            new_attn = _try_instantiate(attn_fn, config, layer_idx, total_layers, old_module)
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.error("Could not replace %s with '%s': %s", name, attention_key, e, exc_info=True)
            failed_count += 1
            raise

        copied_custom = False
        if hasattr(new_attn, "copy_weights") and callable(new_attn.copy_weights):
            try:
                new_attn.copy_weights(old_module)
                copied_custom = True
            except (
                AttributeError,
                KeyError,
                RuntimeError,
                TypeError,
                ValueError,
                OSError,
                ImportError,
            ):
                pass

        if not copied_custom:
            try:
                from neuralops.core._attention_surgery import PROJ_REGEX_PATTERNS, _find_submodule_by_regex, _copy_projection_weights
                old_q = _find_submodule_by_regex(old_module, PROJ_REGEX_PATTERNS["q"])
                old_k = _find_submodule_by_regex(old_module, PROJ_REGEX_PATTERNS["k"])
                old_v = _find_submodule_by_regex(old_module, PROJ_REGEX_PATTERNS["v"])
                old_o = _find_submodule_by_regex(old_module, PROJ_REGEX_PATTERNS["o"])

                new_q = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS["q"])
                new_k = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS["k"])
                new_v = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS["v"])
                new_o = _find_submodule_by_regex(new_attn, PROJ_REGEX_PATTERNS["o"])

                if old_q is not None and new_q is not None and hasattr(old_q, "weight"):
                    _copy_projection_weights(new_q, old_q.weight, getattr(old_q, "bias", None))

                if old_k is not None and new_k is not None and hasattr(old_k, "weight"):
                    w_k = old_k.weight
                    b_k = getattr(old_k, "bias", None)
                    if old_q is not None and hasattr(old_q, "weight") and w_k.shape[0] < old_q.weight.shape[0]:
                        g = old_q.weight.shape[0] // w_k.shape[0]
                        w_k = w_k.repeat_interleave(g, dim=0)
                        if b_k is not None:
                            b_k = b_k.repeat_interleave(g, dim=0)
                    _copy_projection_weights(new_k, w_k, b_k)

                if old_v is not None and new_v is not None and hasattr(old_v, "weight"):
                    w_v = old_v.weight
                    b_v = getattr(old_v, "bias", None)
                    if old_q is not None and hasattr(old_q, "weight") and w_v.shape[0] < old_q.weight.shape[0]:
                        g = old_q.weight.shape[0] // w_v.shape[0]
                        w_v = w_v.repeat_interleave(g, dim=0)
                        if b_v is not None:
                            b_v = b_v.repeat_interleave(g, dim=0)
                    _copy_projection_weights(new_v, w_v, b_v)

                if old_o is not None and new_o is not None and hasattr(old_o, "weight"):
                    _copy_projection_weights(new_o, old_o.weight, getattr(old_o, "bias", None))
            except (AttributeError, RuntimeError, ValueError, KeyError) as e:
                logger.debug("Automatic projection transfer failed for %s: %s", name, e)

        try:
            old_param = next(old_module.parameters())
            new_attn.to(dtype=old_param.dtype, device=old_param.device)
        except (StopIteration, RuntimeError):
            pass

        calib_data = kwargs.get("calibration_data", kwargs.get("calibration_batch", None))
        if calib_data is not None:
            try:
                with torch.no_grad():
                    if isinstance(calib_data, dict):
                        inp = calib_data.get("input_ids", calib_data.get("hidden_states", None))
                    elif isinstance(calib_data, (tuple, list)):
                        inp = calib_data[0]
                    else:
                        inp = calib_data
                    if inp is not None and isinstance(inp, torch.Tensor):
                        inp_dev = inp.to(device=next(old_module.parameters()).device)
                        if inp_dev.dim() == 2 and hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
                            h_in = model.transformer.wte(inp_dev)
                        elif inp_dev.dim() == 3:
                            h_in = inp_dev
                        else:
                            h_in = None
                        if h_in is not None:
                            o_mod = getattr(new_attn, "o_proj", getattr(new_attn, "c_proj", getattr(new_attn, "out_proj", None)))
                            if o_mod is not None and hasattr(o_mod, "weight") and isinstance(o_mod.weight, nn.Parameter):
                                old_out = old_module(h_in)
                                old_y = old_out[0] if isinstance(old_out, tuple) else old_out
                                new_out = new_attn(h_in)
                                new_y = new_out[0] if isinstance(new_out, tuple) else new_out
                                if isinstance(old_y, torch.Tensor) and isinstance(new_y, torch.Tensor) and old_y.shape == new_y.shape:
                                    Y_s = new_y.reshape(-1, new_y.shape[-1]).float()
                                    Y_t = old_y.reshape(-1, old_y.shape[-1]).float()
                                    M = torch.matmul(Y_s.t(), Y_t)
                                    u_p, _, vt_p = torch.linalg.svd(M, full_matrices=False)
                                    R = torch.matmul(u_p, vt_p).to(dtype=o_mod.weight.dtype, device=o_mod.weight.device)
                                    if o_mod.weight.shape == R.shape:
                                        o_mod.weight.data.copy_(torch.matmul(o_mod.weight.data, R))
            except (RuntimeError, ValueError, AttributeError) as calib_err:
                logger.debug("Closed-form Procrustes alignment skipped: %s", calib_err)

        wrapped_attn = UniversalAttentionWrapper(new_attn)

        from neuralops.core.t5_compat import is_t5_model

        if is_t5_model(config):
            wrapped_attn.is_t5 = True
            new_attn.is_t5 = True
            for attr in [
                "has_relative_attention_bias",
                "relative_attention_bias",
                "relative_attention_num_buckets",
                "relative_attention_max_distance",
                "is_decoder",
                "layer_idx",
            ]:
                if hasattr(old_module, attr):
                    val = getattr(old_module, attr)
                    setattr(new_attn, attr, val)
                    setattr(wrapped_attn, attr, val)
            if hasattr(old_module, "compute_bias"):
                new_attn.compute_bias = old_module.compute_bias
                wrapped_attn.compute_bias = old_module.compute_bias

        parts = name.split(".")
        parent_name = ".".join(parts[:-1])
        child_name = parts[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, wrapped_attn)
        replaced_count += 1

    if replaced_count == 0 and failed_count == 0:
        print(f"[WARN] No attention layers found to replace for key='{attention_key}'.")
    elif failed_count > 0:
        print(
            f"Replaced {replaced_count}/{replaced_count + failed_count} layers with '{attention_key}' ({failed_count} skipped)."
        )
    else:
        print(f"Successfully replaced {replaced_count} attention layers with '{attention_key}'.")

    meta_entry = {
        "op": "replace",
        "type": str(attention_key) if attention_key is not None else getattr(new_attention_cls, "__name__", "custom"),
        "kwargs": {
            "layer_indices": layer_indices,
            **kwargs,
        },
    }
    if not hasattr(model, "_neuralops_metadata"):
        model._neuralops_metadata = []
    model._neuralops_metadata.append(meta_entry)

    return model


replace_attention_in_model = replace_attention
