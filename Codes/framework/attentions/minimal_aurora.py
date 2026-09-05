import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
from framework.core.registry import ATTENTION_REGISTRY


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    while cos.dim() < q.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    if cos.shape[-2] != q.shape[-2]:
        cos = cos[..., : q.shape[-2], :]
        sin = sin[..., : q.shape[-2], :]

    rotary_dim = cos.shape[-1]
    head_dim = q.shape[-1]

    if rotary_dim <= head_dim:
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        q_rot_embed = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
        k_rot_embed = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
        q = torch.cat([q_rot_embed, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot_embed
        k = torch.cat([k_rot_embed, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot_embed
    else:
        cos_cut, sin_cut = cos[..., :head_dim], sin[..., :head_dim]
        q = (q * cos_cut.to(q.dtype)) + (_rotate_half(q) * sin_cut.to(q.dtype))
        k = (k * cos_cut.to(k.dtype)) + (_rotate_half(k) * sin_cut.to(k.dtype))

    return q, k


class FactorisedLinear(nn.Module):
    r"""SVD-Factorized Linear Transformation with LoRA Residual Branch."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        energy: float = 0.95,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.energy = energy
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_scaling = (lora_alpha / lora_rank) if lora_rank > 0 else 1.0

        self.down: nn.Linear = nn.Linear(in_features, self.rank, bias=False)
        self.up: nn.Linear = nn.Linear(self.rank, out_features, bias=bias)

        if self.lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, lora_rank))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.down.weight)
        nn.init.orthogonal_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)
        if self.lora_rank > 0 and self.lora_A is not None:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            self.lora_A.requires_grad = True
        if self.lora_rank > 0 and self.lora_B is not None:
            nn.init.zeros_(self.lora_B)
            self.lora_B.requires_grad = True

    def decompose_from_weight(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        energy: Optional[float] = None,
    ) -> None:
        with torch.no_grad():
            W = weight.float()
            # Handle KV dimension matching for GQA
            if W.shape[0] < self.out_features and self.out_features % W.shape[0] == 0:
                rep = self.out_features // W.shape[0]
                W = W.repeat(rep, 1)
                if bias is not None:
                    bias = bias.repeat(rep)
            elif W.shape[0] > self.out_features and W.shape[0] % self.out_features == 0:
                rep = W.shape[0] // self.out_features
                W = W.view(self.out_features, rep, -1).mean(dim=1)
                if bias is not None:
                    bias = bias.view(self.out_features, rep).mean(dim=1)
            elif W.shape[0] != self.out_features or W.shape[1] != self.in_features:
                padded_W = torch.zeros(self.out_features, self.in_features, device=W.device, dtype=W.dtype)
                padded_W[:min(self.out_features, W.shape[0]), :min(self.in_features, W.shape[1])] = W[:min(self.out_features, W.shape[0]), :min(self.in_features, W.shape[1])]
                W = padded_W

            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
            target_energy = energy if energy is not None else self.energy
            total_energy = (S**2).sum()
            cum_energy = torch.cumsum(S**2, dim=0) / (total_energy + 1e-12)
            calc_rank = int((cum_energy < target_energy).sum().item()) + 1
            r = min(self.rank, calc_rank, U.size(1), Vt.size(0))
            r = max(1, r)

            sqrt_s = torch.sqrt(S[:r])
            A = (U[:, :r] * sqrt_s).to(self.up.weight.dtype)
            B = (sqrt_s.unsqueeze(1) * Vt[:r, :]).to(self.down.weight.dtype)

            self.up.weight.data.zero_()
            self.down.weight.data.zero_()
            self.up.weight.data[:, :r] = A
            self.down.weight.data[:r, :] = B

            if bias is not None and self.up.bias is not None:
                self.up.bias.data[:min(len(self.up.bias), len(bias))].copy_(bias[:min(len(self.up.bias), len(bias))])

            self.freeze_base_weights()

    def freeze_base_weights(self) -> None:
        self.down.weight.requires_grad = False
        self.up.weight.requires_grad = False
        if self.up.bias is not None:
            self.up.bias.requires_grad = False
        if self.lora_rank > 0:
            if self.lora_A is not None:
                self.lora_A.requires_grad = True
            if self.lora_B is not None:
                self.lora_B.requires_grad = True

    @property
    def weight(self) -> torch.Tensor:
        base_w = torch.matmul(self.up.weight, self.down.weight)
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            return base_w + (self.lora_B @ self.lora_A) * self.lora_scaling
        return base_w

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.up.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.up(self.down(x))
        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out


@ATTENTION_REGISTRY.register("minimal_aurora", supports_causal=True, supports_gqa=True)
class MinimalAURORAAttention(BaseAttention):
    r"""
    MinimalAURORA: Energy-Preserving SVD-Factorized Attention.
    Hardened for multi-architecture deployment across DistilBERT, RoBERTa, GPT-2, and Qwen3.5.
    """

    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        energy: float = 0.95,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        cfg = model_config if model_config is not None else config
        actual_rank = getattr(
            cfg,
            "svd_rank",
            rank if rank is not None else getattr(cfg, "rank", 64),
        )
        super().__init__(cfg, actual_rank)
        self.num_heads: int = getattr(cfg, "num_attention_heads", getattr(cfg, "num_heads", kwargs.get("num_attention_heads", 4)))
        self.hidden_size: int = getattr(cfg, "hidden_size", kwargs.get("hidden_size", 768))
        self.head_dim: int = self.hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.all_head_size: int = self.num_heads * self.head_dim

        cfg_kv = getattr(cfg, "num_key_value_heads", getattr(cfg, "num_kv_heads", None)) if cfg else None
        if cfg_kv is not None:
            self.num_kv_heads = int(cfg_kv)
        elif kwargs.get("num_kv_heads") is not None:
            self.num_kv_heads = int(kwargs["num_kv_heads"])
        else:
            self.num_kv_heads = self.num_heads

        self.num_kv_heads = max(1, min(self.num_kv_heads, self.num_heads))
        self.kv_repeat: int = self.num_heads // self.num_kv_heads
        self.kv_head_size: int = self.num_kv_heads * self.head_dim

        self.rank: int = actual_rank
        self.energy: float = energy
        self.lora_rank: int = lora_rank
        self.lora_alpha: float = lora_alpha

        if is_decoder is not None:
            self.is_decoder = bool(is_decoder)
        else:
            self.is_decoder = (
                getattr(cfg, "is_decoder", False)
                or getattr(cfg, "causal", False)
                or getattr(cfg, "model_type", "").lower() in ("gpt2", "qwen", "qwen2", "llama")
                if cfg else False
            )

        self.is_nested_encoder: bool = False

        self.q_proj = FactorisedLinear(self.hidden_size, self.all_head_size, rank=self.rank, bias=True, energy=energy, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)
        self.k_proj = FactorisedLinear(self.hidden_size, self.kv_head_size, rank=self.rank, bias=True, energy=energy, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)
        self.v_proj = FactorisedLinear(self.hidden_size, self.kv_head_size, rank=self.rank, bias=True, energy=energy, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)
        self.out_proj = FactorisedLinear(self.all_head_size, self.hidden_size, rank=self.rank, bias=True, energy=energy, lora_rank=self.lora_rank, lora_alpha=self.lora_alpha)

        dropout_p = getattr(cfg, "dropout", getattr(cfg, "attention_probs_dropout_prob", dropout)) if cfg else dropout
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)
        self.pres_gate: nn.Parameter = nn.Parameter(torch.tensor(2.0))

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    @property
    def query(self) -> FactorisedLinear:
        return self.q_proj

    @property
    def key(self) -> FactorisedLinear:
        return self.k_proj

    @property
    def value(self) -> FactorisedLinear:
        return self.v_proj

    def copy_weights(self, src: Any) -> None:
        from framework.core.adapters import UniversalAdapter

        extracted: Dict[str, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = {
            "q": (None, None),
            "k": (None, None),
            "v": (None, None),
            "o": (None, None),
        }

        # 1. UniversalAdapter extraction
        for proj in ["q", "k", "v", "o"]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                if w is not None:
                    extracted[proj] = (w, b)
            except Exception:
                pass

        # 2. GPT-2 Fused c_attn Conv1D handling
        if hasattr(src, "c_attn") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            weight = src.c_attn.weight.data
            bias = getattr(src.c_attn, "bias", None)
            bias_data = bias.data if bias is not None else None
            if weight.dim() == 2 and weight.shape[1] == 3 * self.hidden_size:
                weight = weight.t()
            split_size = weight.shape[0] // 3
            q_w = weight[:split_size, :]
            k_w = weight[split_size : 2 * split_size, :]
            v_w = weight[2 * split_size :, :]
            q_b = bias_data[:split_size] if bias_data is not None else None
            k_b = bias_data[split_size : 2 * split_size] if bias_data is not None else None
            v_b = bias_data[2 * split_size :] if bias_data is not None else None
            extracted["q"] = (q_w, q_b)
            extracted["k"] = (k_w, k_b)
            extracted["v"] = (v_w, v_b)

        if hasattr(src, "c_proj") and extracted["o"][0] is None:
            w_o = src.c_proj.weight.data
            if w_o.dim() == 2 and w_o.shape[0] != self.hidden_size and w_o.shape[1] == self.hidden_size:
                w_o = w_o.t()
            b_o = getattr(src.c_proj, "bias", None)
            extracted["o"] = (w_o, b_o.data if b_o is not None else None)

        # 3. RoBERTa / BERT Nested Encoder handling
        if hasattr(src, "self") and (extracted["q"][0] is None or extracted["k"][0] is None or extracted["v"][0] is None):
            s = getattr(src, "self")
            if hasattr(s, "query"):
                extracted["q"] = (s.query.weight.data, getattr(s.query, "bias", None))
            if hasattr(s, "key"):
                extracted["k"] = (s.key.weight.data, getattr(s.key, "bias", None))
            if hasattr(s, "value"):
                extracted["v"] = (s.value.weight.data, getattr(s.value, "bias", None))

        if hasattr(src, "output") and hasattr(getattr(src, "output"), "dense") and extracted["o"][0] is None:
            dense = src.output.dense
            extracted["o"] = (dense.weight.data, getattr(dense, "bias", None))

        # 4. Modern Decoders (Qwen3.5, LLaMA, Mistral) & DistilBERT
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
                        layer = getattr(src, name)
                        if hasattr(layer, "weight"):
                            extracted[proj] = (layer.weight.data, getattr(layer, "bias", None))
                            break

        # Detect nested encoder architecture
        cls_name = src.__class__.__name__
        has_out_proj = any(hasattr(src, attr) for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o", "proj"))
        self.is_nested_encoder = (
            cls_name in ("BertSelfAttention", "RobertaSelfAttention")
            or (cls_name.endswith("SelfAttention") and not has_out_proj)
        ) and not has_out_proj

        # Check required weights
        missing = [k for k in ["q", "k", "v"] if extracted[k][0] is None]
        if not self.is_nested_encoder and extracted["o"][0] is None:
            missing.append("o")
        if missing:
            raise RuntimeError(f"MinimalAURORAAttention missing required weights for projections: {missing}")

        # Frobenius norm validation
        for proj_key in ["q", "k", "v"]:
            w = extracted[proj_key][0]
            if w is not None:
                norm = torch.linalg.norm(w.float())
                if norm < 1e-4:
                    raise RuntimeError(f"MinimalAURORAAttention extracted weight for {proj_key} has near-zero Frobenius norm ({norm.item():.6f})")

        if extracted["o"][0] is not None:
            norm_o = torch.linalg.norm(extracted["o"][0].float())
            if norm_o < 1e-4:
                raise RuntimeError(f"MinimalAURORAAttention extracted weight for output projection has near-zero Frobenius norm ({norm_o.item():.6f})")

        def _orient(w: torch.Tensor, in_dim: int) -> torch.Tensor:
            if w.dim() == 2 and w.shape[1] != in_dim and w.shape[0] == in_dim:
                return w.t()
            return w

        # Copy into factorized projection modules
        with torch.no_grad():
            for proj_key, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
                w, b = extracted[proj_key]
                if w is not None:
                    in_dim = self.all_head_size if proj_key == "o" else self.hidden_size
                    w_oriented = _orient(w, in_dim)
                    target.decompose_from_weight(w_oriented, b, self.energy)
                elif proj_key == "o" and self.is_nested_encoder:
                    eye_w = torch.eye(self.hidden_size, self.all_head_size, device=self.out_proj.up.weight.device, dtype=self.out_proj.up.weight.dtype)
                    target.decompose_from_weight(eye_w, None, self.energy)

                target.freeze_base_weights()

        if hasattr(src, "is_decoder"):
            self.is_decoder = bool(src.is_decoder)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_kv = kv_input.shape[1]

        Q = self.q_proj(hidden_states).view(B, N, H, D).transpose(1, 2)
        K = self.k_proj(kv_input).view(B, N_kv, self.num_kv_heads, D).transpose(1, 2)
        V = self.v_proj(kv_input).view(B, N_kv, self.num_kv_heads, D).transpose(1, 2)

        if position_embeddings is None:
            position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and not is_cross:
            cos, sin = position_embeddings
            Q, K = apply_rotary_pos_emb(Q, K, cos, sin)

        if self.num_kv_heads < self.num_heads:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(Q.float(), K.float().transpose(-2, -1)) * scale

        if self.is_decoder and not is_cross and N > 1:
            causal_mask = torch.triu(
                torch.full((N, N_kv), float("-inf"), device=device, dtype=torch.float32),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
            scores = scores + causal_mask

        eff_mask = encoder_attention_mask if is_cross else attention_mask
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                scores = scores + (eff_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * 10000.0
            elif eff_mask.dim() == 3:
                scores = scores + eff_mask.unsqueeze(1).float()
            elif eff_mask.dim() == 4:
                scores = scores + eff_mask.float()

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn_weights = F.softmax(scores, dim=-1).to(dtype)
        attn_weights = self.dropout(attn_weights)

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 3:
                head_mask = head_mask.unsqueeze(1)
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, H * D)
        context = torch.nan_to_num(context, nan=0.0, posinf=1e4, neginf=-1e4)

        if self.is_nested_encoder:
            output = context
        else:
            output = self.out_proj(context)

        alpha = torch.sigmoid(self.pres_gate)
        output = alpha * output + (1.0 - alpha) * hidden_states
        return output, (attn_weights if output_attentions else None)

    def get_ortho_loss(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.out_proj.up.weight.device)


@ATTENTION_REGISTRY.register("minimal_aurora_inject")
def inject_minimal_aurora(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 1,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
) -> MinimalAURORAAttention:
    r"""Injection hook for MinimalAURORAAttention into transformer layer."""
    energy = getattr(comp_config, "svd_energy", 0.95) if comp_config else 0.95
    dropout = getattr(comp_config, "dropout", 0.1) if comp_config else 0.1
    new_attn = MinimalAURORAAttention(
        model_config=model_config,
        energy=energy,
        dropout=dropout,
    )
    src_attn = old_attn if old_attn is not None else getattr(layer, "attention", getattr(layer, "self", None))
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception:
            pass
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    if hasattr(layer, "attention"):
        layer.attention = new_attn
    elif hasattr(layer, "self"):
        layer.self = new_attn
    return new_attn
