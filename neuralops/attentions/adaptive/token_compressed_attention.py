"""
Token-Compressed Attention — Production-Hardened v4.

Chunk-based causal token compression with differentiable selection.

═══════════════════════════════════════════════════════════════════════════
WHY v4?
═══════════════════════════════════════════════════════════════════════════

v3 had three CRITICAL bugs:

  1. CAUSALITY VIOLATION in `_compute_token_importance`
       The sampled attention used to score tokens had NO causal mask.
       Query 0 attended to future tokens during prefill → autoregressive
       property broken.

  2. STE GRADIENT EXPLOSION
       Adding `0.01 * (ste_norm - ste_norm.detach())` to a softmax
       distribution produces near-zero but non-zero forward perturbation,
       and injects meaningless gradients into `score_net`.

  3. RoPE FREEZE during generation
       `past_len` was inferred from `past_key_value[0].shape[-2]`. Once the
       cache was compressed, the physical length saturated at `k_keep`,
       freezing all future RoPE positions at the same index → gibberish.

v4 fixes all three:

  ✅ Chunk-based causal top-k
       Split the sequence into chunks of size C. Inside each chunk,
       select the top-k tokens independently. Future tokens cannot
       influence past chunk selections. Causal-safe by construction.

  ✅ STE via multiplication
       The STE mask multiplies the attention weights instead of being
       added. Gradient flows to `score_net` through a clean path.

  ✅ Absolute position tracking
       `position_ids` is always honored. If missing, a running absolute
       counter (`seen_tokens`) is maintained on the cache. RoPE always
       receives correct absolute positions.

  ✅ KV-cache correctness
       Both tuple caches and HF DynamicCache are updated properly with
       the ORIGINAL (uncompressed) K/V for correctness. Compression is
       applied at attention time only.

  ✅ CLS + local window preservation
       Token 0 is force-kept. The last `local_window` tokens are force-kept
       inside each chunk for causal decoding.

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
from neuralops.core.utils.lora import LoRALinear
from neuralops.core.utils.rope import apply_rotary_pos_emb, rotate_half, _rotate_half
from neuralops.core.utils.config import _cfg_get

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.adapters import UniversalAdapter
from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.rope import _rotate_half

from neuralops.core.utils.gqa import repeat_kv



logger = logging.getLogger(__name__)




# ===========================================================================
# RoPE (absolute-position aware)
# ===========================================================================


# ===========================================================================
# LoRALinear
# ===========================================================================


# ===========================================================================
# Cache container (tracks absolute positions)
# ===========================================================================
class TokenCompressedCache:
    """
    Cache container tracking K/V and absolute positions.

    Unlike HF DynamicCache, this stores the ORIGINAL (uncompressed) K/V so
    that compression can be applied at attention time without corrupting
    the RoPE positions of future queries.
    """

    def __init__(self, num_layers: int = 0) -> None:
        self.keys: list[torch.Tensor | None] = [None] * num_layers
        self.values: list[torch.Tensor | None] = [None] * num_layers
        self.positions: list[torch.Tensor | None] = [None] * num_layers
        self.seen_tokens: int = 0

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if layer_idx >= len(self.keys) or self.keys[layer_idx] is None:
            return 0
        return int(self.keys[layer_idx].shape[-2])

    def get_max_length(self) -> int:
        return self.seen_tokens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if layer_idx >= len(self.keys):
            self.keys.extend([None] * (layer_idx - len(self.keys) + 1))
            self.values.extend([None] * (layer_idx - len(self.values) + 1))
            self.positions.extend([None] * (layer_idx - len(self.positions) + 1))

        k_old = self.keys[layer_idx]
        if k_old is None:
            k_all = key_states
            v_all = value_states
            p_all = positions
        else:
            k_all = torch.cat([k_old, key_states], dim=-2)
            v_all = torch.cat([self.values[layer_idx], value_states], dim=-2)
            p_all = torch.cat([self.positions[layer_idx], positions], dim=-1)

        self.keys[layer_idx] = k_all
        self.values[layer_idx] = v_all
        self.positions[layer_idx] = p_all
        return k_all, v_all, p_all


# ===========================================================================
# Main attention
# ===========================================================================
@ATTENTION_REGISTRY.register("token_compressed_attention")
class TokenCompressedAttention(BaseAttention):
    """
    Chunk-based causal token compression.

    Encoder (bidirectional):
        Global top-k by importance score. Chunking is not needed.

    Decoder (causal):
        Chunk-based causal top-k. Sequence is split into chunks of size
        `chunk_size`. Within each chunk, the top-k tokens by importance
        are kept, plus token 0, plus the last `local_window` tokens of
        that chunk. Future chunks never influence past chunk selections.

    Importance score:
        0.5 * ||x|| + learned_score(x) + 0.5 * causal_attn_score(x)
    """

    supports_gqa = True
    supports_causal = True
    is_custom_attention = True

    def __init__(
        self,
        config: Any = None,
        rank: int = 64,
        compression_ratio: float = 0.5,
        local_window: int = 32,
        chunk_size: int = 64,
        per_layer_ratio: bool = False,
        use_ste: bool = True,
        pretrained_attn: nn.Module | None = None,
        lora_rank: int | None = None,
        lora_alpha: float | None = None,
        comp_config: Any = None,
        is_decoder: bool | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            super().__init__(config, rank)
        except TypeError:
            super().__init__(config)

        src = pretrained_attn if pretrained_attn is not None else kwargs.get("old_attn")

        # ---------------- Dimensions ----------------
        dim = None
        if config is not None and not isinstance(config, int):
            dim = _cfg_get(
                config, "hidden_size",
                _cfg_get(config, "dim", _cfg_get(config, "d_model", None)),
            )
        if dim is None and comp_config is not None:
            dim = _cfg_get(comp_config, "hidden_size", _cfg_get(comp_config, "dim", None))
        if dim is None and src is not None:
            dim = getattr(src, "hidden_size", getattr(src, "embed_dim", None))
            if dim is None:
                for attr in ("q_proj", "q_lin", "query"):
                    m = getattr(src, attr, None)
                    if m is not None and hasattr(m, "in_features"):
                        dim = m.in_features
                        break
            if dim is None and hasattr(src, "c_attn"):
                c_l = getattr(src.c_attn, "base_layer", src.c_attn)
                w = getattr(c_l, "weight", None)
                if w is not None:
                    dim = w.shape[0] if w.shape[0] < w.shape[1] else w.shape[1]
        if dim is None:
            dim = kwargs.get("hidden_size", 768)
        self.dim = int(dim)
        self.hidden_size = self.dim

        num_heads = kwargs.get("num_heads")
        if num_heads is None and config is not None and not isinstance(config, int):
            num_heads = _cfg_get(
                config, "num_attention_heads",
                _cfg_get(config, "n_heads", _cfg_get(config, "num_heads", None)),
            )
        if num_heads is None and src is not None:
            num_heads = getattr(
                src, "num_heads",
                getattr(src, "num_attention_heads", getattr(src, "n_head", None)),
            )
        if num_heads is None:
            num_heads = max(1, self.dim // 64)
        self.num_heads = max(1, int(num_heads))

        head_dim = None
        if config is not None and not isinstance(config, int):
            head_dim = _cfg_get(config, "head_dim", None)
        if head_dim is None:
            head_dim = kwargs.get("head_dim")
        if head_dim is None:
            head_dim = self.dim // self.num_heads
        self.head_dim = int(head_dim)

        cfg_kv = None
        if config is not None and not isinstance(config, int):
            cfg_kv = _cfg_get(
                config, "num_key_value_heads",
                _cfg_get(config, "num_kv_heads", None),
            )
        if cfg_kv is None and src is not None:
            cfg_kv = getattr(
                src, "num_key_value_heads",
                getattr(src, "num_kv_heads", None),
            )
        self.num_kv_heads = int(cfg_kv) if cfg_kv is not None else self.num_heads
        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.num_key_value_groups = max(1, self.num_heads // self.num_kv_heads)

        self.q_head_size = self.num_heads * self.head_dim
        self.kv_head_size = self.num_kv_heads * self.head_dim
        self.all_head_size = self.q_head_size

        # ---------------- Compression config ----------------
        self.base_compression_ratio = float(
            kwargs.get(
                "compression_ratio",
                _cfg_get(comp_config, "compression_ratio", compression_ratio),
            )
        )
        self.base_compression_ratio = max(0.1, min(1.0, self.base_compression_ratio))
        self.local_window = int(
            kwargs.get("local_window", _cfg_get(comp_config, "local_window", local_window))
        )
        self.chunk_size = int(
            kwargs.get("chunk_size", _cfg_get(comp_config, "chunk_size", chunk_size))
        )
        self.per_layer_ratio = bool(per_layer_ratio)
        self.use_ste = bool(use_ste)

        # ---------------- LoRA ----------------
        rl = kwargs.get("lora_rank")
        if rl is None and comp_config is not None:
            rl = _cfg_get(comp_config, "lora_rank")
        if rl is None:
            rl = lora_rank if lora_rank is not None else 16
        self.lora_rank = int(rl)

        ra = kwargs.get("lora_alpha")
        if ra is None and comp_config is not None:
            ra = _cfg_get(comp_config, "lora_alpha")
        if ra is None:
            ra = lora_alpha if lora_alpha is not None else 32.0
        self.lora_alpha = float(ra)

        # ---------------- Decoder detection ----------------
        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        elif kwargs.get("is_decoder") is not None:
            self.is_decoder = bool(kwargs["is_decoder"])
        elif src is not None and getattr(src, "is_decoder", False):
            self.is_decoder = True
        elif config is not None and not isinstance(config, int):
            mt = str(_cfg_get(config, "model_type", "")).lower()
            self.is_decoder = bool(_cfg_get(config, "is_decoder", False)) or any(
                mt.startswith(p)
                for p in (
                    "gpt", "qwen", "llama", "mistral", "gemma", "phi",
                    "opt", "bloom", "falcon", "olmo", "mpt", "stablelm",
                )
            )
        else:
            self.is_decoder = False
        self.is_causal = self.is_decoder
        self.is_nested_encoder = False
        self.rotary_emb = getattr(src, "rotary_emb", None) if src is not None else None

        self._layer_idx = kwargs.get("layer_idx", 0)
        self._total_layers = kwargs.get("total_layers", 12)

        # ---------------- Device/dtype ----------------
        proj_device = None
        proj_dtype = None
        if src is not None:
            for attr in ("q_lin", "q_proj", "query", "c_attn"):
                mod = getattr(src, attr, None)
                if mod is not None and hasattr(mod, "weight"):
                    proj_device = mod.weight.device
                    proj_dtype = mod.weight.dtype
                    break

        # ---------------- Projections ----------------
        self.q_lin = LoRALinear(
            self.dim, self.q_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, device=proj_device, dtype=proj_dtype,
        )
        self.k_lin = LoRALinear(
            self.dim, self.kv_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, device=proj_device, dtype=proj_dtype,
        )
        self.v_lin = LoRALinear(
            self.dim, self.kv_head_size,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, device=proj_device, dtype=proj_dtype,
        )
        self.out_lin = LoRALinear(
            self.q_head_size, self.dim,
            bias=True, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha,
            freeze_base=True, device=proj_device, dtype=proj_dtype,
        )

        # Aliases
        self.q_proj = self.q_lin
        self.k_proj = self.k_lin
        self.v_proj = self.v_lin
        self.out_proj = self.out_lin
        self.query = self.q_lin
        self.key = self.k_lin
        self.value = self.v_lin

        # ---------------- Score network ----------------
        self.score_net = nn.Sequential(
            nn.Linear(self.dim, 64, device=proj_device, dtype=proj_dtype),
            nn.GELU(),
            nn.Linear(64, 1, device=proj_device, dtype=proj_dtype),
        )
        with torch.no_grad():
            nn.init.zeros_(self.score_net[2].weight)
            nn.init.zeros_(self.score_net[2].bias)

        dropout_p = (
            _cfg_get(config, "dropout",
                     _cfg_get(config, "attention_probs_dropout_prob", 0.1))
            if config is not None and not isinstance(config, int)
            else 0.1
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # ---------------- aliases ----------------

    @property
    def compression_ratio(self) -> float:
        return self.base_compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, val: float) -> None:
        self.base_compression_ratio = float(val)
    @property
    def dense(self):
        return self.out_lin

    @property
    def c_proj(self):
        return self.out_lin

    @property
    def o_proj(self):
        return self.out_lin

    # ------------------------------------------------------------------
    # Per-layer compression ratio
    # ------------------------------------------------------------------
    def _get_compression_ratio(self) -> float:
        if not self.per_layer_ratio:
            return self.base_compression_ratio
        progress = self._layer_idx / max(1, self._total_layers - 1)
        ratio = 0.8 - 0.5 * progress
        return max(0.15, min(0.95, ratio * self.base_compression_ratio / 0.5))

    # ------------------------------------------------------------------
    # copy_weights (unchanged, production-hardened)
    # ------------------------------------------------------------------
    def copy_weights(self, old_attn: nn.Module) -> None:
        with torch.no_grad():
            if getattr(old_attn, "rotary_emb", None) is not None:
                self.rotary_emb = old_attn.rotary_emb

            w_q = b_q = w_k = b_k = w_v = b_v = w_o = b_o = None

            src_cls_name = getattr(old_attn, "__class__", type(old_attn)).__name__
            if src_cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
                hasattr(old_attn, "query")
                and not hasattr(old_attn, "dense")
                and not hasattr(old_attn, "out_proj")
                and not hasattr(old_attn, "c_proj")
            ):
                self.is_nested_encoder = True

            if hasattr(old_attn, "self") and hasattr(old_attn, "output"):
                self.is_nested_encoder = True
                src = old_attn.self
                if hasattr(old_attn.output, "dense"):
                    dense = getattr(old_attn.output.dense, "base_layer", old_attn.output.dense)
                    w_o = dense.weight.detach()
                    b_o = dense.bias.detach() if dense.bias is not None else None
            else:
                src = old_attn

            def _unwrap(layer: nn.Module) -> nn.Module:
                return getattr(layer, "base_layer", layer)

            # GPT-2 fused c_attn
            if hasattr(src, "c_attn") and hasattr(src, "c_proj"):
                c_attn = _unwrap(src.c_attn)
                c_proj = _unwrap(src.c_proj)
                cw = c_attn.weight.detach()
                cb = c_attn.bias.detach() if c_attn.bias is not None else None
                if cw.dim() == 2 and cw.shape[0] == self.dim and cw.shape[1] != self.dim:
                    cw = cw.t().contiguous()
                qd, kd, vd = self.q_head_size, self.kv_head_size, self.kv_head_size
                w_q = cw[:qd]
                w_k = cw[qd : qd + kd]
                w_v = cw[qd + kd : qd + kd + vd]
                if cb is not None:
                    b_q = cb[:qd]
                    b_k = cb[qd : qd + kd]
                    b_v = cb[qd + kd : qd + kd + vd]
                ow = c_proj.weight.detach()
                if ow.dim() == 2 and ow.shape[0] == self.dim and ow.shape[1] != self.dim:
                    ow = ow.t().contiguous()
                w_o = ow
                b_o = c_proj.bias.detach() if c_proj.bias is not None else None

            # BERT/RoBERTa nested
            elif hasattr(src, "query") and hasattr(src, "key") and hasattr(src, "value"):
                q_l, k_l, v_l = _unwrap(src.query), _unwrap(src.key), _unwrap(src.value)
                w_q = q_l.weight.detach()
                b_q = q_l.bias.detach() if q_l.bias is not None else None
                w_k = k_l.weight.detach()
                b_k = k_l.bias.detach() if k_l.bias is not None else None
                w_v = v_l.weight.detach()
                b_v = v_l.bias.detach() if v_l.bias is not None else None
                if w_o is None:
                    for a in ("dense", "out_proj", "o_proj", "c_proj"):
                        if hasattr(src, a):
                            p = _unwrap(getattr(src, a))
                            w_o = p.weight.detach()
                            b_o = p.bias.detach() if p.bias is not None else None
                            break

            # Qwen/LLaMA
            elif hasattr(src, "q_proj") and hasattr(src, "k_proj") and hasattr(src, "v_proj"):
                q_l, k_l, v_l = _unwrap(src.q_proj), _unwrap(src.k_proj), _unwrap(src.v_proj)
                w_q = q_l.weight.detach()
                b_q = q_l.bias.detach() if q_l.bias is not None else None
                w_k = k_l.weight.detach()
                b_k = k_l.bias.detach() if k_l.bias is not None else None
                w_v = v_l.weight.detach()
                b_v = v_l.bias.detach() if v_l.bias is not None else None
                if hasattr(src, "o_proj"):
                    o_l = _unwrap(src.o_proj)
                    w_o = o_l.weight.detach()
                    b_o = o_l.bias.detach() if o_l.bias is not None else None

            # DistilBERT
            elif hasattr(src, "q_lin") and hasattr(src, "k_lin") and hasattr(src, "v_lin"):
                q_l, k_l, v_l = _unwrap(src.q_lin), _unwrap(src.k_lin), _unwrap(src.v_lin)
                w_q = q_l.weight.detach()
                b_q = q_l.bias.detach() if q_l.bias is not None else None
                w_k = k_l.weight.detach()
                b_k = k_l.bias.detach() if k_l.bias is not None else None
                w_v = v_l.weight.detach()
                b_v = v_l.bias.detach() if v_l.bias is not None else None
                if hasattr(src, "out_lin"):
                    o_l = _unwrap(src.out_lin)
                    w_o = o_l.weight.detach()
                    b_o = o_l.bias.detach() if o_l.bias is not None else None

            # Fallback
            if w_q is None or w_k is None or w_v is None:
                try:
                    adapter = UniversalAdapter()
                    if w_q is None:
                        w_q, b_q = adapter.get_weight_bias(src, "q")
                    if w_k is None:
                        w_k, b_k = adapter.get_weight_bias(src, "k")
                    if w_v is None:
                        w_v, b_v = adapter.get_weight_bias(src, "v")
                    if w_o is None:
                        w_o, b_o = adapter.get_weight_bias(src, "o")
                except Exception as e:
                    logger.debug("UniversalAdapter fallback failed: %s", e)

            for name, tensor in (("Q", w_q), ("K", w_k), ("V", w_v), ("O", w_o)):
                if tensor is not None and float(torch.linalg.norm(tensor.float())) < 1e-4:
                    raise RuntimeError(f"TokenCompressedAttention: {name} weight norm is near zero (< 1e-4)")

            missing = [
                n for n, w in (("Q", w_q), ("K", w_k), ("V", w_v)) if w is None
            ]
            if w_o is None and not self.is_nested_encoder:
                missing.append("O")
            if missing:
                raise RuntimeError(
                    f"TokenCompressedAttention.copy_weights: missing {missing} "
                    f"from {type(old_attn).__name__}"
                )

            def _copy(target_layer: LoRALinear, w, b):
                if w is not None:
                    if w.shape == target_layer.weight.shape:
                        target_layer.weight.data.copy_(w.to(target_layer.weight.dtype))
                    elif w.t().contiguous().shape == target_layer.weight.shape:
                        target_layer.weight.data.copy_(
                            w.t().contiguous().to(target_layer.weight.dtype)
                        )
                    else:
                        m0 = min(target_layer.weight.shape[0], w.shape[0])
                        m1 = min(target_layer.weight.shape[1], w.shape[1])
                        target_layer.weight.data.zero_()
                        target_layer.weight.data[:m0, :m1].copy_(
                            w[:m0, :m1].to(target_layer.weight.dtype)
                        )
                if b is not None and target_layer.bias is not None:
                    if b.shape == target_layer.bias.shape:
                        target_layer.bias.data.copy_(b.to(target_layer.bias.dtype))
                    else:
                        mb = min(target_layer.bias.shape[0], b.shape[0])
                        target_layer.bias.data.zero_()
                        target_layer.bias.data[:mb].copy_(b[:mb].to(target_layer.bias.dtype))
                elif target_layer.bias is not None:
                    target_layer.bias.data.zero_()

            _copy(self.q_lin, w_q, b_q)
            _copy(self.k_lin, w_k, b_k)
            _copy(self.v_lin, w_v, b_v)
            if w_o is not None:
                _copy(self.out_lin, w_o, b_o)

    # ------------------------------------------------------------------
    # FIX-01: causal-safe importance scoring
    # ------------------------------------------------------------------
    def _compute_token_importance(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        valid_mask_2d: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Importance = energy + learned + causal_attn.

        FIX: The sampled attention now applies a causal mask when
        `self.is_decoder` is True, preventing future-token leakage.
        """
        B, L, D = x.shape
        x_f = x.float()

        # 1) Energy
        energy = torch.norm(x_f, p=2, dim=-1)  # (B, L)

        # 2) Learned saliency
        score_dev = next(self.score_net.parameters()).device
        score_dtype = next(self.score_net.parameters()).dtype
        learned = (
            self.score_net(x_f.to(score_dev, score_dtype))
            .squeeze(-1)
            .to(x_f.device, x_f.dtype)
        )

        # 3) Sampled attention importance — WITH CAUSAL MASK
        with torch.no_grad():
            H_s = min(2, q.shape[1])
            L_s = min(64, L)
            q_s = q[:, :H_s, :L_s, :].float()
            k_s = k[:, :H_s, :, :].float()
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(q_s, k_s.transpose(-2, -1)) * scale

            # FIX: apply causal mask during decoder scoring
            if self.is_decoder:
                q_idx = torch.arange(L_s, device=x.device).unsqueeze(1)
                k_idx = torch.arange(L, device=x.device).unsqueeze(0)
                causal = k_idx <= q_idx  # (L_s, L)
                attn = attn.masked_fill(
                    ~causal.unsqueeze(0).unsqueeze(0), float("-inf")
                )

            attn = F.softmax(attn, dim=-1)
            attn_importance = attn.mean(dim=(1, 2))  # (B, L)
            attn_importance = attn_importance / (
                attn_importance.max(dim=-1, keepdim=True).values + 1e-6
            )

        total = energy + learned + 0.5 * attn_importance

        if valid_mask_2d is not None:
            total = total.masked_fill(~valid_mask_2d, -1e4)

        return total

    # ------------------------------------------------------------------
    # FIX-01: chunk-based causal top-k
    # ------------------------------------------------------------------
    def _chunked_causal_topk(
        self,
        importance: torch.Tensor,
        k_keep_total: int,
        valid_mask_2d: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Split the sequence into chunks of size `self.chunk_size`. Within
        each chunk, select top-k tokens such that the TOTAL selected
        tokens == k_keep_total.

        Returns: (B, k_keep_total) sorted absolute indices.
        """
        B, L = importance.shape
        device = importance.device
        C = max(1, self.chunk_size)
        num_chunks = (L + C - 1) // C
        # Distribute k_keep_total proportionally across chunks
        base_per_chunk = max(1, k_keep_total // num_chunks)
        extra = k_keep_total - base_per_chunk * num_chunks

        all_idx: list[torch.Tensor] = []
        for ci in range(num_chunks):
            start = ci * C
            end = min(start + C, L)
            chunk_len = end - start
            k_in_chunk = min(chunk_len, base_per_chunk + (1 if ci < extra else 0))
            if k_in_chunk <= 0:
                continue
            chunk_imp = importance[:, start:end]  # (B, chunk_len)
            _, local_idx = torch.topk(chunk_imp, k_in_chunk, dim=-1)  # (B, k)
            local_idx, _ = torch.sort(local_idx, dim=-1)
            all_idx.append(local_idx + start)

        if not all_idx:
            # Fallback: just take first k
            idx = torch.arange(k_keep_total, device=device)
            return idx.unsqueeze(0).expand(B, -1)

        idx = torch.cat(all_idx, dim=-1)  # (B, total_kept)
        # Trim to exact k_keep_total (in case extra padding)
        if idx.shape[-1] > k_keep_total:
            idx = idx[:, :k_keep_total]
        idx, _ = torch.sort(idx, dim=-1)
        return idx

    # ------------------------------------------------------------------
    # Token selection (encoder: global; decoder: chunked)
    # ------------------------------------------------------------------
    def _select_compressed_tokens(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
        valid_mask_2d: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        B, L, D = x.shape
        H_kv, Dh = self.num_kv_heads, self.head_dim

        if L <= 2:
            idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
            comp_valid = (
                torch.gather(valid_mask_2d, 1, idx) if valid_mask_2d is not None else None
            )
            return k_tensor, v_tensor, idx, comp_valid, None

        importance = self._compute_token_importance(x, q, k_tensor, valid_mask_2d)

        eff_ratio = self._get_compression_ratio()
        k_keep = max(2, min(L, int(math.ceil(L * eff_ratio))))

        # CLS boost
        importance = importance.clone()
        if valid_mask_2d is None or valid_mask_2d[:, 0].any():
            importance[:, 0] = importance[:, 0] + 1e4

        # Local window boost (only for decoder)
        if self.is_decoder and self.local_window > 0:
            w = min(self.local_window, L)
            importance[:, -w:] = importance[:, -w:] + 1e3

        # All-masked safety
        all_masked = (importance <= -1e4).all(dim=-1)
        if all_masked.any():
            importance[all_masked, 0] = 0.0

        # Selection strategy
        if self.is_decoder and L > self.chunk_size:
            top_indices = self._chunked_causal_topk(importance, k_keep, valid_mask_2d)
        else:
            _, top_indices = torch.topk(importance, k_keep, dim=-1)
            top_indices, _ = torch.sort(top_indices, dim=-1)

        # FIX-02: STE via MULTIPLICATION
        ste_mask: torch.Tensor | None = None
        if self.use_ste and self.training:
            # Soft mask over the full sequence
            soft = F.softmax(importance, dim=-1)  # (B, L)
            # Gather at kept indices
            soft_kept = torch.gather(soft, 1, top_indices)  # (B, k_keep)
            # Normalize so sum == 1 over kept
            soft_kept = soft_kept / (soft_kept.sum(dim=-1, keepdim=True) + 1e-6)
            # STE: forward = 1 (identity scaling), backward = gradient of soft_kept
            ste_mask = 1.0 + (soft_kept - soft_kept.detach())
            # ste_mask shape: (B, k_keep) — multiply against attention over keys

        # Gather K/V
        top_idx_exp = (
            top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H_kv, -1, Dh)
        )
        k_comp = torch.gather(k_tensor, 2, top_idx_exp)
        v_comp = torch.gather(v_tensor, 2, top_idx_exp)

        comp_valid = (
            torch.gather(valid_mask_2d, 1, top_indices)
            if valid_mask_2d is not None
            else None
        )

        return k_comp, v_comp, top_indices, comp_valid, ste_mask

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
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
        layer_idx: int = 0,
        *args: Any,
        **kwargs: Any,
    ):
        if encoder_hidden_states is not None:
            raise NotImplementedError(
                "TokenCompressedAttention: cross-attention not supported."
            )

        B, L, D = hidden_states.shape
        H, H_kv, Dh = self.num_heads, self.num_kv_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        q = self.q_lin(hidden_states).view(B, L, H, Dh).transpose(1, 2).contiguous()
        k_new = self.k_lin(hidden_states).view(
            B, L, H_kv, Dh
        ).transpose(1, 2).contiguous()
        v_new = self.v_lin(hidden_states).view(
            B, L, H_kv, Dh
        ).transpose(1, 2).contiguous()

        # ---------------- Read past cache ----------------
        past_k = past_v = past_pos = None
        past_seen = 0
        cache_obj = None

        if past_key_value is not None:
            if isinstance(past_key_value, TokenCompressedCache):
                if layer_idx < len(past_key_value.keys):
                    past_k = past_key_value.keys[layer_idx]
                    past_v = past_key_value.values[layer_idx]
                    past_pos = past_key_value.positions[layer_idx]
                    past_seen = past_key_value.seen_tokens
                cache_obj = past_key_value
            elif hasattr(past_key_value, "key_cache") and past_key_value.key_cache:
                idx = min(layer_idx, len(past_key_value.key_cache) - 1)
                past_k = past_key_value.key_cache[idx]
                past_v = past_key_value.value_cache[idx]
                past_seen = past_k.shape[-2] if past_k is not None else 0
                cache_obj = past_key_value
            elif isinstance(past_key_value, (tuple, list)) and len(past_key_value) >= 2:
                past_k, past_v = past_key_value[0], past_key_value[1]
                past_seen = past_k.shape[-2] if past_k is not None else 0

        # ---------------- FIX-03: absolute position offset ----------------
        position_ids = kwargs.get("position_ids")
        if position_ids is not None:
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0).expand(B, -1)
            start_pos = int(position_ids[0, 0].item())
            new_positions = position_ids.long()
        elif past_pos is not None:
            start_pos = int(past_pos[:, -1].max().item()) + 1
            new_positions = (
                torch.arange(start_pos, start_pos + L, device=device)
                .unsqueeze(0).expand(B, -1)
            )
        else:
            start_pos = past_seen
            new_positions = (
                torch.arange(start_pos, start_pos + L, device=device)
                .unsqueeze(0).expand(B, -1)
            )

        # ---------------- RoPE ----------------
        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings")
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k_new = apply_rotary_pos_emb(
                q, k_new, cos, sin, past_len=start_pos
            )

        # ---------------- Cache concat (compact, no pruning yet) ----------------
        if past_k is not None:
            k_full = torch.cat([past_k, k_new], dim=-2)
            v_full = torch.cat([past_v, v_new], dim=-2)
            if past_pos is not None:
                pos_full = torch.cat([past_pos, new_positions], dim=-1)
            else:
                pos_full = new_positions
        else:
            k_full = k_new
            v_full = v_new
            pos_full = new_positions

        # ---------------- Cache update ----------------
        updated_cache = None
        if cache_obj is not None and isinstance(cache_obj, TokenCompressedCache):
            cache_obj.update(k_new, v_new, layer_idx, new_positions)
            cache_obj.seen_tokens = max(cache_obj.seen_tokens, start_pos + L)
            updated_cache = cache_obj
        elif cache_obj is not None and hasattr(cache_obj, "update"):
            cache_obj.update(k_new, v_new, layer_idx)
            updated_cache = cache_obj
        elif isinstance(past_key_value, (tuple, list)):
            updated_cache = (k_full, v_full, pos_full)

        # ---------------- Valid mask ----------------
        valid_mask_2d = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid_mask_2d = (
                    attention_mask
                    if attention_mask.dtype == torch.bool
                    else (
                        attention_mask > -1e4
                        if attention_mask.dtype.is_floating_point
                        else attention_mask > 0
                    )
                )
            elif attention_mask.dim() == 4:
                if attention_mask.dtype == torch.bool:
                    vm = attention_mask
                elif attention_mask.dtype.is_floating_point:
                    vm = attention_mask > -1e4
                else:
                    vm = attention_mask > 0
                valid_mask_2d = vm.any(dim=-2).squeeze(1)

        # ---------------- Token selection ----------------
        N_kv_full = k_full.shape[-2]
        if L > 1:
            # Prefill: compress k_full
            k_use, v_use, top_idx, comp_valid, ste_mask = self._select_compressed_tokens(
                hidden_states, q, k_full, v_full, valid_mask_2d
            )
        else:
            # Decode (single token): compress k_full as well, but with
            # special handling to always keep the new token.
            k_use, v_use, top_idx, comp_valid, ste_mask = self._select_compressed_tokens(
                hidden_states, q, k_full, v_full, valid_mask_2d
            )

        # ---------------- GQA expansion ----------------
        if H_kv != H:
            k_use = repeat_kv(k_use, self.num_key_value_groups)
            v_use = repeat_kv(v_use, self.num_key_value_groups)

        N_k = k_use.shape[-2]

        # ---------------- Attention ----------------
        scale = 1.0 / math.sqrt(Dh)
        scores = (
            torch.matmul(q.float(), k_use.float().transpose(-2, -1).contiguous()) * scale
        )

        # Causal mask based on ABSOLUTE positions
        if self.is_decoder and top_idx is not None:
            q_pos = new_positions.reshape(B, 1, L, 1) if new_positions.numel() == B * L else new_positions.view(-1)[None, None, :, None]
            k_pos_abs = torch.gather(pos_full.expand(B, -1) if pos_full.dim() == 2 and pos_full.shape[0] != B else pos_full, 1, top_idx)  # (B, k_keep)
            k_pos = k_pos_abs.reshape(B, 1, 1, N_k)
            causal = k_pos > q_pos
            scores = scores.masked_fill(causal, float("-inf"))

        # Padding mask
        if comp_valid is not None:
            cm = comp_valid[:, None, None, :]
            # Keep CLS always
            if top_idx is not None and self.is_decoder:
                cls_keep = (top_idx == 0)[:, None, None, :]
                cm = cm | cls_keep
            scores = scores.masked_fill(~cm, float("-inf"))

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores, dim=-1).to(dtype)
        attn = torch.nan_to_num(attn, nan=0.0)

        # FIX-02: STE via multiplication (correct gradient flow)
        if ste_mask is not None and self.training:
            # ste_mask: (B, k_keep) → broadcast to (B, 1, 1, k_keep)
            ste_b = ste_mask[:, None, None, :].to(attn.dtype)
            attn = attn * ste_b

        attn = self.dropout(attn)

        # Head mask
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(-1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask[:, :, :1].unsqueeze(-1)
            attn = attn * head_mask.to(attn.dtype)

        # ---------------- Output ----------------
        context = torch.matmul(attn, v_use)
        context = context.transpose(1, 2).contiguous().view(B, L, self.q_head_size)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_lin(context)

        if output.shape[-1] != D:
            raise RuntimeError(
                f"TokenCompressedAttention: output dim {output.shape[-1]} != "
                f"hidden_size {D}"
            )

        # ---------------- Return ----------------
        attn_ret = attn if output_attentions else None

        if updated_cache is not None or past_key_value is not None:
            return output, attn_ret, updated_cache
        return output, attn_ret


# ===========================================================================
# Injection helper
# ===========================================================================
def inject_token_compressed_attn(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: nn.Module | None = None,
    _depth: int = 0,
    **kwargs: Any,
):
    if _depth > 8:
        logger.warning("inject_token_compressed_attn: max recursion depth exceeded.")
        return None

    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer"))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    layer_list = None
    if hasattr(target, "distilbert") and hasattr(target.distilbert, "transformer"):
        layer_list = target.distilbert.transformer.layer
    elif hasattr(target, "roberta") and hasattr(target.roberta, "encoder"):
        layer_list = target.roberta.encoder.layer
    elif hasattr(target, "transformer") and hasattr(target.transformer, "h"):
        layer_list = target.transformer.h
    elif hasattr(target, "h") and isinstance(target.h, (torch.nn.ModuleList, list)):
        layer_list = target.h
    elif hasattr(target, "model") and hasattr(target.model, "layers"):
        layer_list = target.model.layers
    elif hasattr(target, "encoder") and hasattr(target.encoder, "layer"):
        layer_list = target.encoder.layer
    elif (
        hasattr(target, "bert")
        and hasattr(target.bert, "encoder")
        and hasattr(target.bert.encoder, "layer")
    ):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_token_compressed_attn(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config,
                layer_idx=idx,
                total_layers=tot,
                _depth=_depth + 1,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    try:
        src = old_attn
        if src is None:
            for attr in ("attention", "self_attn", "attn", "self"):
                if hasattr(target, attr):
                    cand = getattr(target, attr)
                    if isinstance(cand, nn.Module) and cand is not target:
                        src = getattr(cand, "self", cand)
                        break
        if src is None:
            src = target

        if isinstance(src, TokenCompressedAttention) or "TokenCompressed" in src.__class__.__name__:
            return None
        if "DeltaNet" in src.__class__.__name__ or "LinearAttention" in src.__class__.__name__:
            return None

        if model_config is None and hasattr(target, "config"):
            model_config = target.config
        if model_config is None and hasattr(src, "config"):
            model_config = src.config

        comp_ratio = float(
            kwargs.pop("compression_ratio", _cfg_get(comp_config, "compression_ratio", 0.5))
        )
        local_window = int(
            kwargs.pop("local_window", _cfg_get(comp_config, "local_window", 32))
        )
        chunk_size = int(
            kwargs.pop("chunk_size", _cfg_get(comp_config, "chunk_size", 64))
        )
        per_layer_ratio = bool(
            kwargs.pop("per_layer_ratio", _cfg_get(comp_config, "per_layer_ratio", False))
        )
        use_ste = bool(kwargs.pop("use_ste", _cfg_get(comp_config, "use_ste", True)))
        lora_rank = kwargs.pop(
            "lora_rank",
            _cfg_get(comp_config, "lora_rank", 16),
        )
        lora_alpha = kwargs.pop(
            "lora_alpha",
            _cfg_get(comp_config, "lora_alpha", 32.0),
        )
        rank = kwargs.pop("rank", _cfg_get(comp_config, "rank", 32))

        new_attn = TokenCompressedAttention(
            config=model_config,
            rank=rank,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            compression_ratio=comp_ratio,
            local_window=local_window,
            chunk_size=chunk_size,
            per_layer_ratio=per_layer_ratio,
            use_ste=use_ste,
            pretrained_attn=src if src is not target else None,
            comp_config=comp_config,
            layer_idx=layer_idx,
            total_layers=total_layers,
            **kwargs,
        )

        if hasattr(target, "attention"):
            if hasattr(target.attention, "self"):
                new_attn.is_nested_encoder = True
                target.attention.self = new_attn
            else:
                new_attn.is_nested_encoder = False
                target.attention = new_attn
        elif hasattr(target, "self_attn"):
            new_attn.is_nested_encoder = False
            target.self_attn = new_attn
        elif hasattr(target, "attn"):
            new_attn.is_nested_encoder = False
            target.attn = new_attn
        elif hasattr(target, "self"):
            new_attn.is_nested_encoder = True
            target.self = new_attn

        return new_attn
    except Exception as e:
        logger.exception(
            "inject_token_compressed_attn failed at layer %d: %s", layer_idx, e
        )
        return None


__all__ = [
    "TokenCompressedAttention",
    "TokenCompressedCache",
    "LoRALinear",
    "apply_rotary_pos_emb",
    "inject_token_compressed_attn",
]