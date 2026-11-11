#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
ClusterFormer: Token Clustering Sparse Attention

Clusters query tokens into semantic centroids to approximate full pairwise self-attention in linear time.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.attention_mask import get_token_valid_mask
from neuralops.core.attention_utils import LoRALinear, apply_rotary_pos_emb
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.utils.gqa import repeat_kv
from neuralops.utils.model_family import ModelAdapter

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register(
    "cluster_attention",
    supports_causal=True,
    supports_bidirectional=True,
    supports_gqa=True,
)
class ClusterAttention(BaseAttention):
    """ClusterAttention neural operator.

    ClusterFormer: Token Clustering Attention Module.

    Attributes:
        supports_causal (bool): Whether causal autoregressive masking is supported.
        is_custom_attention (bool): Flag indicating specialized non-standard attention mechanics.
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True

    def __init__(
        self,
        config: Any | None = None,
        rank: int | None = None,
        num_clusters: int = 16,
        n_iters: int = 3,
        window_size: int = 32,
        lora_rank: int = 16,
        comp_config: Any | None = None,
        pretrained_attn: nn.Module | None = None,
        old_attn: nn.Module | None = None,
        **kwargs: Any,
    ) -> None:
        source_attn = pretrained_attn if pretrained_attn is not None else old_attn
        super().__init__(config, rank)

        hidden_size = kwargs.get("d_model") or kwargs.get("hidden_size") or kwargs.get("dim")
        num_heads = kwargs.get("num_heads") or kwargs.get("n_heads") or kwargs.get("num_attention_heads")
        num_kv_heads = kwargs.get("num_kv_heads") or kwargs.get("num_key_value_heads")

        if not num_heads and source_attn is not None:
            num_heads = getattr(
                source_attn,
                "num_heads",
                getattr(
                    source_attn,
                    "num_attention_heads",
                    getattr(source_attn, "n_heads", None),
                ),
            )
        if not num_heads and config is not None:
            num_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", None))
        num_heads = int(num_heads or 12)

        if not hidden_size and source_attn is not None:
            cand = getattr(
                source_attn,
                "hidden_size",
                getattr(
                    source_attn,
                    "embed_dim",
                    getattr(source_attn, "all_head_size", None),
                ),
            )
            if cand is None and hasattr(source_attn, "query") and hasattr(source_attn.query, "in_features"):
                cand = source_attn.query.in_features
            if cand is None and hasattr(source_attn, "q_proj") and hasattr(source_attn.q_proj, "in_features"):
                cand = source_attn.q_proj.in_features
            if cand is None and hasattr(source_attn, "q_lin") and hasattr(source_attn.q_lin, "in_features"):
                cand = source_attn.q_lin.in_features
            if cand is None and hasattr(source_attn, "c_attn") and hasattr(source_attn.c_attn, "in_features"):
                cand = source_attn.c_attn.in_features
            hidden_size = cand
        if not hidden_size and config is not None:
            hidden_size = getattr(config, "hidden_size", getattr(config, "dim", None))
        hidden_size = int(hidden_size or 768)

        head_dim = kwargs.get("head_dim", 0)
        if not head_dim and source_attn is not None:
            head_dim = getattr(source_attn, "head_dim", 0)
            if not head_dim:
                for attr in ("q_proj", "query", "q_lin"):
                    mod = getattr(source_attn, attr, None)
                    if mod is not None and hasattr(mod, "weight"):
                        base_mod = getattr(mod, "base_layer", mod)
                        w = base_mod.weight
                        is_conv = "Conv1D" in type(base_mod).__name__ or hasattr(base_mod, "nf")
                        out_dim = w.shape[1] if (is_conv and w.ndim == 2) else w.shape[0]
                        head_dim = out_dim // max(1, num_heads)
                        break

        if not head_dim and config is not None:
            head_dim = getattr(config, "head_dim", 0)
            if not head_dim:
                try:
                    adapter = ModelAdapter(config)
                    head_dim = getattr(adapter, "head_dim", 0)
                except Exception:
                    # allowed-swallow: best-effort-telemetry
                    pass

        if not head_dim:
            head_dim = hidden_size // max(1, num_heads) if num_heads > 0 else 64
        head_dim = int(head_dim)

        if not num_kv_heads and source_attn is not None:
            num_kv_heads = getattr(source_attn, "num_key_value_heads", None) or getattr(
                source_attn, "num_kv_heads", None
            )
            if not num_kv_heads:
                for attr in ("k_proj", "key", "k_lin"):
                    mod = getattr(source_attn, attr, None)
                    if mod is not None and hasattr(mod, "weight"):
                        base_mod = getattr(mod, "base_layer", mod)
                        w = base_mod.weight
                        is_conv = "Conv1D" in type(base_mod).__name__ or hasattr(base_mod, "nf")
                        out_dim = w.shape[1] if (is_conv and w.ndim == 2) else w.shape[0]
                        num_kv_heads = out_dim // max(1, head_dim)
                        break

        if not num_kv_heads and config is not None:
            num_kv_heads = getattr(config, "num_key_value_heads", None)

        num_kv_heads = int(num_kv_heads or num_heads)

        self.hidden_size: int = hidden_size
        self.d_model: int = hidden_size
        self.num_heads: int = num_heads
        self.num_kv_heads: int = num_kv_heads
        self.head_dim: int = head_dim
        self.all_head_size: int = self.num_heads * self.head_dim
        self.kv_head_size: int = self.num_kv_heads * self.head_dim
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))

        if config is not None:
            self.num_clusters = getattr(config, "cluster_num_clusters", num_clusters)
        else:
            self.num_clusters = num_clusters
        self.n_iters: int = max(1, n_iters)
        self.window_size: int = window_size
        self.temp = nn.Parameter(torch.tensor(0.5))
        self.local_residual = nn.Parameter(torch.tensor(5.5))

        self.register_buffer("_seq_idx", torch.arange(4096), persistent=False)

        model_type = str(getattr(config, "model_type", "") or "").lower() if config else ""
        _decoder_prefixes = (
            "gpt",
            "qwen",
            "llama",
            "mistral",
            "gemma",
            "phi",
            "opt",
            "bloom",
            "falcon",
            "olmo",
            "mpt",
            "stablelm",
        )
        is_dec = (
            (
                getattr(config, "is_decoder", False)
                or getattr(config, "is_causal", False)
                or getattr(config, "causal", False)
                or any(model_type.startswith(p) for p in _decoder_prefixes)
            )
            if config
            else False
        )
        if hasattr(config, "architectures") and any(
            "GPT" in str(a) or "CausalLM" in str(a) for a in (getattr(config, "architectures", []) or [])
        ):
            is_dec = True
        if source_attn is not None:
            cls_name = source_attn.__class__.__name__.lower()
            if (
                any(p in cls_name for p in ("gpt", "causal", "qwen", "llama", "mistral"))
                or getattr(source_attn, "is_causal", False)
                or getattr(source_attn, "is_decoder", False)
                or hasattr(source_attn, "c_attn")
            ):
                is_dec = True
        if kwargs.get("is_decoder") is not None:
            is_dec = bool(kwargs["is_decoder"])
        elif kwargs.get("is_causal") is not None:
            is_dec = bool(kwargs["is_causal"])
        elif kwargs.get("causal") is not None:
            is_dec = bool(kwargs["causal"])
        self.is_decoder: bool = bool(is_dec)
        self.is_causal: bool = bool(is_dec)
        self.is_nested_encoder: bool = bool(kwargs.get("is_nested_encoder", False))

        if config is not None:
            self.lora_rank = getattr(
                config,
                "lora_rank",
                getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank)),
            )
            self.lora_alpha = getattr(
                config,
                "lora_alpha",
                getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0)),
            )
        elif comp_config is not None:
            self.lora_rank = getattr(comp_config, "lora_rank", kwargs.get("lora_rank", lora_rank))
            self.lora_alpha = getattr(comp_config, "lora_alpha", kwargs.get("lora_alpha", 32.0))
        else:
            self.lora_rank = kwargs.get("lora_rank", lora_rank)
            self.lora_alpha = kwargs.get("lora_alpha", 32.0)
        self.freeze_base: bool = self.lora_rank > 0

        self.q_proj = LoRALinear(
            self.hidden_size,
            self.all_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.k_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.v_proj = LoRALinear(
            self.hidden_size,
            self.kv_head_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )
        self.out_proj = LoRALinear(
            self.all_head_size,
            self.hidden_size,
            bias=True,
            lora_rank=self.lora_rank,
            lora_alpha=self.lora_alpha,
            freeze_base=self.freeze_base,
        )

        dropout_p = (
            getattr(
                config,
                "dropout",
                getattr(
                    config,
                    "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", 0.1),
                ),
            )
            if config
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)
        self.rotary_emb = getattr(source_attn, "rotary_emb", None) if source_attn is not None else None

        if source_attn is not None:
            self.copy_weights(source_attn, **kwargs)

    def copy_weights(self, src: Any, **kwargs: Any) -> None:
        from neuralops.core.adapters import UniversalAdapter

        extracted: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception as e:
                logger.debug(f"UniversalAdapter extraction failed for {proj}: {e}")

        if (
            hasattr(src, "c_attn")
            and src.c_attn is not None
            and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None)
        ):
            c_attn_mod = src.c_attn
            if hasattr(c_attn_mod, "merge") and not getattr(c_attn_mod, "merged", False):
                try:
                    c_attn_mod.merge()
                except Exception:
                    # allowed-swallow: best-effort-telemetry
                    pass
            base_c = getattr(c_attn_mod, "base_layer", c_attn_mod)
            weight = getattr(base_c, "weight", getattr(c_attn_mod, "weight", None))
            if weight is not None:
                weight = weight.data.clone()
                p_cls = getattr(base_c, "__class__", type(base_c)).__name__
                if (
                    ("Conv1D" in p_cls)
                    or hasattr(base_c, "nf")
                    or ("Conv1D" in getattr(c_attn_mod, "__class__", type(c_attn_mod)).__name__)
                    or hasattr(c_attn_mod, "nf")
                    or (weight.ndim == 2 and weight.shape[1] == 3 * self.hidden_size)
                ):
                    weight = weight.t().contiguous()
                split_size = weight.shape[0] // 3
                bias_mod = getattr(base_c, "bias", getattr(c_attn_mod, "bias", None))
                bias_data = bias_mod.data.clone() if bias_mod is not None else None
                extracted["q"] = (
                    weight[:split_size, :],
                    bias_data[:split_size] if bias_data is not None else None,
                )
                extracted["k"] = (
                    weight[split_size : 2 * split_size, :],
                    (bias_data[split_size : 2 * split_size] if bias_data is not None else None),
                )
                extracted["v"] = (
                    weight[2 * split_size :, :],
                    bias_data[2 * split_size :] if bias_data is not None else None,
                )

        if hasattr(src, "c_proj") and src.c_proj is not None and extracted["o"][0] is None:
            c_proj_mod = src.c_proj
            if hasattr(c_proj_mod, "merge") and not getattr(c_proj_mod, "merged", False):
                try:
                    c_proj_mod.merge()
                except Exception:
                    # allowed-swallow: best-effort-telemetry
                    pass
            base_o = getattr(c_proj_mod, "base_layer", c_proj_mod)
            w_o = getattr(base_o, "weight", getattr(c_proj_mod, "weight", None))
            if w_o is not None:
                w_o = w_o.data.clone()
                p_cls = getattr(base_o, "__class__", type(base_o)).__name__
                if (
                    ("Conv1D" in p_cls)
                    or hasattr(base_o, "nf")
                    or ("Conv1D" in getattr(c_proj_mod, "__class__", type(c_proj_mod)).__name__)
                    or hasattr(c_proj_mod, "nf")
                    or (w_o.ndim == 2 and w_o.shape[0] < w_o.shape[1])
                ):
                    w_o = w_o.t().contiguous()
                bias_o = getattr(base_o, "bias", getattr(c_proj_mod, "bias", None))
                extracted["o"] = (
                    w_o,
                    bias_o.data.clone() if bias_o is not None else None,
                )

        def _extract_mod_wb(mod: Any) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if mod is None:
                return None, None
            if hasattr(mod, "merge") and not getattr(mod, "merged", False):
                try:
                    mod.merge()
                except Exception:
                    # allowed-swallow: best-effort-telemetry
                    pass
            base_m = getattr(mod, "base_layer", mod)
            w_m = getattr(base_m, "weight", getattr(mod, "weight", None))
            if w_m is None or not isinstance(w_m, torch.Tensor) or w_m.ndim != 2:
                return None, None
            w_m = w_m.data.clone()
            cls_m = type(mod).__name__
            base_cls_m = type(base_m).__name__
            if ("Conv1D" in cls_m) or ("Conv1D" in base_cls_m) or hasattr(base_m, "nf") or hasattr(mod, "nf"):
                w_m = w_m.t().contiguous()
            b_mod = getattr(base_m, "bias", getattr(mod, "bias", None))
            b_m = b_mod.data.clone() if isinstance(b_mod, torch.Tensor) else None
            return w_m, b_m

        if hasattr(src, "self") and (
            extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None
        ):
            s = src.self
            for key, attr in (("q", "query"), ("k", "key"), ("v", "value")):
                if extracted[key][0] is None and hasattr(s, attr):
                    w_m, b_m = _extract_mod_wb(getattr(s, attr))
                    if w_m is not None:
                        extracted[key] = (w_m, b_m)

        if hasattr(src, "output") and hasattr(src.output, "dense") and extracted["o"][0] is None:
            w_m, b_m = _extract_mod_wb(src.output.dense)
            if w_m is not None:
                extracted["o"] = (w_m, b_m)

        named_map = {
            "q": ["q_proj", "q_lin", "query"],
            "k": ["k_proj", "k_lin", "key"],
            "v": ["v_proj", "v_lin", "value"],
            "o": ["o_proj", "out_proj", "out_lin", "dense"],
        }
        for proj, names in named_map.items():
            if extracted[proj][0] is None:
                for name in names:
                    if hasattr(src, name):
                        w_m, b_m = _extract_mod_wb(getattr(src, name))
                        if w_m is not None:
                            extracted[proj] = (w_m, b_m)
                            break

        cls_name = src.__class__.__name__
        has_out_proj = any(
            hasattr(src, attr)
            for attr in (
                "dense",
                "out_proj",
                "c_proj",
                "out_lin",
                "o_proj",
                "wo",
                "o",
                "proj",
            )
        )
        is_nested = False
        if (
            hasattr(src, "self")
            and hasattr(src.self, "query")
            or (
                cls_name
                in (
                    "BertSelfAttention",
                    "RobertaSelfAttention",
                    "XLMRobertaSelfAttention",
                )
                or (cls_name.endswith("SelfAttention") and not has_out_proj)
                or (("roberta" in cls_name.lower() or "bert" in cls_name.lower()) and not has_out_proj)
            )
        ):
            is_nested = True
        if hasattr(src, "config"):
            if getattr(src.config, "model_type", "").lower() in ("roberta", "bert", "xlm-roberta") and not has_out_proj:
                is_nested = True
        if kwargs.get("is_nested_encoder") is not None:
            is_nested = kwargs["is_nested_encoder"]
        self.is_nested_encoder = bool(is_nested)

        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"ClusterAttention missing required weights for projections: {missing}")

        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    raise RuntimeError(
                        f"ClusterAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})"
                    )

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                raise RuntimeError(
                    f"ClusterAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})"
                )

        with torch.no_grad():
            for proj_key, target in [
                ("q", self.q_proj),
                ("k", self.k_proj),
                ("v", self.v_proj),
                ("o", self.out_proj),
            ]:
                w, b = extracted[proj_key]
                if w is not None:
                    target_w = target.weight
                    out_d, in_d = target_w.shape
                    if w.ndim != 2:
                        w = w.reshape(w.size(0), -1)
                    if w.shape != (out_d, in_d) and w.t().shape == (out_d, in_d):
                        w = w.t().contiguous()
                    elif w.shape[1] != in_d and w.shape[0] == in_d:
                        w = w.t().contiguous()

                    b_val = (b.data if isinstance(b, nn.Parameter) else b) if b is not None else None
                    if proj_key in ("k", "v") and w.shape[0] != out_d:
                        hd = self.head_dim
                        src_out = w.shape[0]
                        if src_out > out_d and src_out % out_d == 0 and out_d % hd == 0:
                            rep = src_out // out_d
                            w = w.view(self.num_kv_heads, rep, hd, -1).mean(dim=1).reshape(out_d, -1)
                            if b_val is not None and b_val.numel() == src_out:
                                b_val = b_val.view(self.num_kv_heads, rep, hd).mean(dim=1).reshape(out_d)
                        elif src_out < out_d and out_d % src_out == 0 and src_out % hd == 0:
                            rep = out_d // src_out
                            src_kv = src_out // hd
                            w = w.view(src_kv, hd, -1).repeat_interleave(rep, dim=0).reshape(out_d, -1)
                            if b_val is not None and b_val.numel() == src_out:
                                b_val = b_val.view(src_kv, hd).repeat_interleave(rep, dim=0).reshape(out_d)

                    src_out, src_in = w.shape
                    target_w.data.zero_()
                    target_w.data[: min(out_d, src_out), : min(in_d, src_in)].copy_(
                        w[: min(out_d, src_out), : min(in_d, src_in)].to(device=target_w.device, dtype=target_w.dtype)
                    )
                    if target.bias is not None:
                        if b_val is not None:
                            target.bias.data.zero_()
                            target.bias.data[: min(len(target.bias), len(b_val))].copy_(
                                b_val[: min(len(target.bias), len(b_val))].to(
                                    device=target.bias.device, dtype=target.bias.dtype
                                )
                            )
                        else:
                            target.bias.data.zero_()
                elif proj_key == "o" and self.is_nested_encoder:
                    min_d = min(target.weight.shape[0], target.weight.shape[1])
                    target.weight.data.zero_()
                    target.weight.data[:min_d, :min_d].fill_diagonal_(1.0)
                    if target.bias is not None:
                        target.bias.data.zero_()

                target.freeze_base_weights()

        if hasattr(src, "rotary_emb") and src.rotary_emb is not None:
            self.rotary_emb = src.rotary_emb

        if bool(getattr(src, "is_decoder", False)) or bool(getattr(src, "is_causal", False)):
            self.is_decoder = True
            self.is_causal = True

    def _kmeans_cluster(
        self,
        Q: torch.Tensor,
        K_num: int,
        iters: int = 3,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Vectorized Lloyd's $K$-means clustering for queries with padding token filtering.
        Q: [B, H, L, d], valid_mask: Optional[torch.Tensor] of shape [B, L] (True for real tokens)
        -> centroids: [B, H, K, d], cluster_ids: [B, H, L], counts: [B, H, K, 1]
        """
        B, H, L, d = Q.shape
        K = min(K_num, L)

        if valid_mask is not None:
            valid_f = valid_mask.float()
            num_valid = valid_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            step = (num_valid / K).clamp(min=1.0)
            k_idx = torch.arange(K, device=Q.device, dtype=torch.float32).unsqueeze(0)
            target_ranks = (k_idx * step).clamp(max=num_valid - 1.0).long()

            ranks = (valid_f.cumsum(dim=-1) - 1.0) * valid_f
            is_valid_3d = valid_mask.unsqueeze(1)
            rank_diff = torch.where(
                is_valid_3d,
                (ranks.unsqueeze(1) - target_ranks.unsqueeze(-1)).abs(),
                torch.tensor(1e6, device=Q.device),
            )
            chosen_indices = torch.argmin(rank_diff, dim=-1)

            idx_expanded = chosen_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, K, d)
            centroids = torch.gather(Q, 2, idx_expanded).clone()
        else:
            step = max(1, L // K)
            init_indices = torch.arange(0, L, step, device=Q.device)[:K]
            if len(init_indices) < K:
                pad_idx = torch.zeros(K - len(init_indices), dtype=torch.long, device=Q.device)
                init_indices = torch.cat([init_indices, pad_idx])
            centroids = Q[:, :, init_indices, :].clone()

        cluster_ids = torch.zeros(B, H, L, dtype=torch.long, device=Q.device)
        counts = torch.ones(B, H, K, 1, dtype=Q.dtype, device=Q.device)
        Q_f = Q.float()

        from neuralops.ops.triton.sparse_attn import (
            fused_kmeans_centroid_assign_pytorch,
            fused_kmeans_centroid_assign_triton,
        )

        for _ in range(iters):
            try:
                centroids, cluster_ids, counts = fused_kmeans_centroid_assign_triton(Q, centroids, valid_mask)
            except Exception:
                centroids, cluster_ids, counts = fused_kmeans_centroid_assign_pytorch(Q, centroids, valid_mask)

        return centroids, cluster_ids, counts

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        head_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        past_key_value: Any | None = None,
        output_attentions: bool | None = False,
        output_hidden_states: bool | None = False,
        return_dict: bool | None = True,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        key_value_states = kwargs.get("key_value_states", encoder_hidden_states)
        is_cross_attention = key_value_states is not None
        kv_input = key_value_states if is_cross_attention else hidden_states

        B, L, D = hidden_states.shape
        N_k = kv_input.shape[1]
        H, Dh = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, Dh).transpose(1, 2).contiguous()

        pos_emb = position_embeddings if position_embeddings is not None else kwargs.get("position_embeddings")
        if pos_emb is not None and isinstance(pos_emb, (tuple, list)) and len(pos_emb) >= 2:
            cos, sin = pos_emb[0], pos_emb[1]
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin, kwargs.get("position_ids"))
        elif getattr(self, "rotary_emb", None) is not None:
            rotary_mod = self.rotary_emb
            pos_ids = kwargs.get("position_ids")
            if pos_ids is None:
                pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
            try:
                cos, sin = rotary_mod(Q, seq_len=L)
            except Exception:
                try:
                    cos, sin = rotary_mod(Q, pos_ids)
                except Exception:
                    try:
                        cos, sin = rotary_mod(Q)
                    except Exception:
                        cos, sin = None, None
            if cos is not None and sin is not None:
                Q, K = apply_rotary_pos_emb(Q, K, cos, sin, position_ids=pos_ids)

        if K.shape[1] != Q.shape[1]:
            if Q.shape[1] % K.shape[1] == 0:
                rep = Q.shape[1] // K.shape[1]
                K = repeat_kv(K, rep)
                V = repeat_kv(V, rep)
            else:
                rep = math.ceil(Q.shape[1] / K.shape[1])
                K = repeat_kv(K, rep)[:, : Q.shape[1], :, :]
                V = repeat_kv(V, rep)[:, : Q.shape[1], :, :]

        scale = 1.0 / math.sqrt(int(Dh))

        mask = attention_mask if attention_mask is not None else kwargs.get("mask", kwargs.get("attention_mask"))
        is_causal_mask = False
        if mask is not None and mask.dim() == 4 and mask.shape[-1] > 1:
            if mask.dtype == torch.bool:
                is_causal_mask = bool((~mask[..., 0, -1]).any().item())
            elif mask.is_floating_point():
                is_causal_mask = bool((mask[..., 0, -1] < -1.0).any().item())

        _cfg = self.config
        is_dec_cfg = False
        if _cfg is not None:
            model_type = str(getattr(_cfg, "model_type", "") or "").lower()
            _decoder_prefixes = (
                "gpt",
                "qwen",
                "llama",
                "mistral",
                "gemma",
                "phi",
                "opt",
                "bloom",
                "falcon",
                "olmo",
                "mpt",
                "stablelm",
            )
            is_dec_cfg = (
                getattr(_cfg, "is_decoder", False)
                or getattr(_cfg, "is_causal", False)
                or getattr(_cfg, "causal", False)
                or any(model_type.startswith(p) for p in _decoder_prefixes)
            )
        is_decoder = bool(
            self.is_decoder
            or getattr(self, "is_causal", False)
            or is_dec_cfg
            or kwargs.get("is_causal", False)
            or kwargs.get("is_decoder", False)
            or kwargs.get("causal", False)
            or is_causal_mask
        )

        if is_cross_attention or is_decoder:
            # Always accumulate Q@K^T and softmax in FP32 to prevent FP16 AMP overflow on outlier activations.
            scores = torch.matmul(Q.float(), K.float().transpose(-2, -1).contiguous()) * scale
            invalid_mask = torch.zeros((B, 1, L, N_k), device=device, dtype=torch.bool)
            if mask is not None:
                if mask.dim() == 4:
                    if mask.dtype == torch.bool:
                        m_bool = ~mask
                    elif mask.dtype.is_floating_point:
                        if (mask < 0).any():
                            m_bool = mask < -1e3
                            soft_bias = mask.float().masked_fill(m_bool, 0.0)
                            if soft_bias.abs().max() > 0:
                                scores = scores + soft_bias
                        else:
                            m_bool = mask <= 0.5
                    else:
                        m_bool = mask <= 0
                    invalid_mask = invalid_mask | m_bool.to(device=device)
                else:
                    valid_m = get_token_valid_mask(mask, B, N_k, device)
                    if valid_m is not None:
                        invalid_mask = invalid_mask | (~valid_m).unsqueeze(1).unsqueeze(2)

            if is_decoder and not is_cross_attention:
                past_len = max(0, N_k - L)
                q_idx = torch.arange(L, device=device).unsqueeze(1)
                k_idx = torch.arange(N_k, device=device).unsqueeze(0)
                causal_invalid = k_idx > (q_idx + past_len)
                invalid_mask = invalid_mask | causal_invalid.unsqueeze(0).unsqueeze(0)

            all_masked = invalid_mask.all(dim=-1, keepdim=True)
            safe_invalid = invalid_mask.clone()
            if all_masked.any():
                safe_invalid[..., :1] = safe_invalid[..., :1] & (~all_masked)

            f32_min = -1e9
            scores_safe = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4).masked_fill(safe_invalid, f32_min)
            attn_exact = F.softmax(scores_safe, dim=-1).masked_fill(all_masked, 0.0)

            def _apply_head_mask(t: torch.Tensor) -> torch.Tensor:
                if head_mask is None:
                    return t
                hm = head_mask.to(device=t.device, dtype=t.dtype)
                if hm.dim() == 1:
                    hm = hm.view(1, -1, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(-1)
                return t * hm

            attn_drop = _apply_head_mask(self.dropout(attn_exact))
            V_f = V.float()
            out_exact = torch.matmul(attn_drop, V_f)

            if is_decoder and not is_cross_attention:
                # Wire temp & local_residual strictly causally so all parameters train cleanly with <1e-5 init MSE.
                temp_scale = 1.0 + 0.1 * torch.tanh(self.temp.float())
                attn_temp = F.softmax(scores_safe * temp_scale, dim=-1).masked_fill(all_masked, 0.0)
                attn_temp_drop = _apply_head_mask(self.dropout(attn_temp))
                out_temp = torch.matmul(attn_temp_drop, V_f)
                alpha = torch.sigmoid(self.local_residual.float())
                out_context = (alpha * out_exact + (1.0 - alpha) * out_temp).to(dtype)
            else:
                out_context = out_exact.to(dtype)

            if not is_decoder:
                valid_q = get_token_valid_mask(attention_mask, B, L, device)
                if valid_q is not None:
                    if valid_q.shape[-1] > L:
                        valid_q = valid_q[:, -L:]
                    out_context = out_context * valid_q.unsqueeze(1).unsqueeze(-1).to(dtype)
            attn_ret = attn_exact.to(dtype) if output_attentions else None
        else:
            valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)
            if valid_mask_2d is not None and valid_mask_2d.shape[-1] > L:
                valid_mask_2d = valid_mask_2d[:, -L:]
            key_valid_4d = (
                valid_mask_2d.unsqueeze(1).unsqueeze(2)
                if valid_mask_2d is not None
                else torch.ones((B, 1, 1, L), dtype=torch.bool, device=device)
            )

            K_num = min(max(1, self.num_clusters), L)

            centroids, cluster_ids, counts = self._kmeans_cluster(
                Q.detach(), K_num, self.n_iters, valid_mask=valid_mask_2d
            )
            centroids_f = torch.nan_to_num(centroids.float(), nan=0.0).detach()
            Q_f = Q.float()
            K_f = K.float()
            V_f = V.float()

            scores_cent = torch.matmul(centroids_f, K_f.transpose(-2, -1).contiguous()) * scale
            scores_cent = scores_cent.clamp(min=-1e4, max=1e4).masked_fill(~key_valid_4d, -1e9)
            all_cent_masked = (~key_valid_4d).all(dim=-1, keepdim=True)
            if all_cent_masked.any():
                scores_cent = scores_cent.masked_fill(all_cent_masked, 0.0)
            attn_cent = F.softmax(scores_cent, dim=-1).masked_fill(all_cent_masked, 0.0)
            attn_cent = self.dropout(attn_cent)

            V_cent = torch.nan_to_num(torch.matmul(attn_cent, V_f), nan=0.0)

            scores_assign = torch.matmul(Q_f, centroids_f.transpose(-2, -1).contiguous()) * scale
            empty_clusters = (counts == 0).squeeze(-1).unsqueeze(2)
            all_empty = empty_clusters.all(dim=-1, keepdim=True)
            if all_empty.any():
                empty_clusters = empty_clusters & (~all_empty)
            scores_assign = scores_assign.clamp(min=-1e4, max=1e4).masked_fill(empty_clusters, -1e9)
            temp_clamped = 0.05 + F.softplus(self.temp.float())
            attn_assign = F.softmax(scores_assign / temp_clamped, dim=-1).masked_fill(all_empty, 0.0)
            global_context = torch.nan_to_num(torch.matmul(attn_assign, V_cent), nan=0.0)

            if self.window_size > 0 and self.window_size < L:
                scores_loc = torch.matmul(Q_f, K_f.transpose(-2, -1).contiguous()) * scale
                idx = (
                    self._seq_idx[:L].to(device)
                    if hasattr(self, "_seq_idx") and len(self._seq_idx) >= L
                    else torch.arange(L, device=device)
                )
                band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
                band_mask_4d = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d
                all_loc_masked = (~band_mask_4d).all(dim=-1, keepdim=True)
                safe_loc_mask = band_mask_4d.clone()
                if all_loc_masked.any():
                    safe_loc_mask[..., :1] = safe_loc_mask[..., :1] | all_loc_masked
                scores_loc = torch.nan_to_num(scores_loc, nan=0.0, posinf=1e4, neginf=-1e4).masked_fill(
                    ~safe_loc_mask, -1e9
                )
                attn_loc = F.softmax(scores_loc, dim=-1).masked_fill(all_loc_masked, 0.0)
                attn_loc = self.dropout(attn_loc)
                local_context = torch.matmul(attn_loc, V_f)
                alpha = torch.sigmoid(self.local_residual.float())
                out_context = ((1.0 - alpha) * global_context + alpha * local_context).to(dtype)
            else:
                out_context = global_context.to(dtype)

            attn_ret = attn_cent.to(dtype) if output_attentions else None

            if head_mask is not None:
                hm = head_mask.to(device=out_context.device, dtype=out_context.dtype)
                if hm.dim() == 1:
                    hm = hm.view(1, -1, 1, 1)
                elif hm.dim() == 2:
                    hm = hm.view(hm.shape[0], hm.shape[1], 1, 1)
                elif hm.dim() == 3:
                    hm = hm.unsqueeze(-1)
                out_context = out_context * hm

            if valid_mask_2d is not None:
                out_context = out_context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(dtype)

        out = out_context.transpose(1, 2).contiguous().view(B, L, self.all_head_size)

        if self.is_nested_encoder:
            output = out
        else:
            output = self.out_proj(out)

        valid_out = get_token_valid_mask(attention_mask, B, L, device)
        if valid_out is not None:
            if valid_out.shape[-1] > L:
                valid_out = valid_out[:, -L:]
            if valid_out.shape[-1] == L:
                output = output * valid_out.unsqueeze(-1).to(output.dtype)

        return output, attn_ret


def inject_cluster(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    **kwargs: Any,
) -> ClusterAttention | nn.Module | None:
    r"""Safe injection factory for ClusterAttention across diverse transformer architectures."""
    target = layer if layer is not None else model
    if target is None:
        return None

    cfg = model_config
    if cfg is None and hasattr(target, "config"):
        cfg = target.config

    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_cluster(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    src_attn = old_attn
    if src_attn is None and target is not None:
        for attr in ("attention", "self_attn", "attn", "self"):
            if hasattr(target, attr):
                cand = getattr(target, attr)
                if hasattr(cand, "self"):
                    src_attn = cand.self
                else:
                    src_attn = cand
                break

    if src_attn is None and target is not None and hasattr(target, "forward"):
        src_attn = target

    if src_attn is None and cfg is None:
        return None

    hidden_size = (
        getattr(src_attn, "hidden_size", None)
        or getattr(src_attn, "embed_dim", None)
        or getattr(src_attn, "dim", None)
        or getattr(src_attn, "all_head_size", None)
        or (
            src_attn.query.in_features
            if hasattr(src_attn, "query") and hasattr(src_attn.query, "in_features")
            else None
        )
        or (
            src_attn.q_proj.in_features
            if hasattr(src_attn, "q_proj") and hasattr(src_attn.q_proj, "in_features")
            else None
        )
        or (
            src_attn.q_lin.in_features
            if hasattr(src_attn, "q_lin") and hasattr(src_attn.q_lin, "in_features")
            else None
        )
        or (
            src_attn.c_attn.in_features
            if hasattr(src_attn, "c_attn") and hasattr(src_attn.c_attn, "in_features")
            else None
        )
        or (getattr(target, "in_features", None) if target is not None else None)
        or (getattr(cfg, "hidden_size", None) if cfg is not None else None)
        or (getattr(cfg, "dim", None) if cfg is not None else None)
        or 768
    )
    num_heads = (
        getattr(src_attn, "num_heads", None)
        or getattr(src_attn, "num_attention_heads", None)
        or getattr(src_attn, "n_heads", None)
        or (getattr(cfg, "num_attention_heads", None) if cfg is not None else None)
        or (getattr(cfg, "n_heads", None) if cfg is not None else None)
        or 12
    )

    num_clusters = getattr(
        comp_config,
        "num_clusters",
        getattr(comp_config, "cluster_num_clusters", kwargs.get("num_clusters", 16)),
    )
    n_iters = getattr(comp_config, "n_iters", kwargs.get("n_iters", 3))
    window_size = getattr(comp_config, "window_size", kwargs.get("window_size", 32))

    try:
        new_block = ClusterAttention(
            config=cfg,
            comp_config=comp_config,
            d_model=hidden_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_clusters=num_clusters,
            n_iters=n_iters,
            window_size=window_size,
            pretrained_attn=src_attn,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to instantiate ClusterAttention: {e}")
        return None

    if target is not None:
        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_block.is_nested_encoder = True
                target.attention.self = new_block
            else:
                new_block.is_nested_encoder = False
                target.attention = new_block
        elif hasattr(target, "self"):
            new_block.is_nested_encoder = True
            target.self = new_block
        elif hasattr(target, "attn"):
            new_block.is_nested_encoder = False
            new_block.is_decoder = True
            new_block.is_causal = True
            target.attn = new_block
        elif hasattr(target, "self_attn"):
            new_block.is_nested_encoder = False
            new_block.is_decoder = True
            new_block.is_causal = True
            target.self_attn = new_block

    return new_block
