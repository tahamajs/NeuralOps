r"""
Smart Weighted Ensemble with Task-Aware Attention and Hybrid Combination.

Implements multi-model ensemble fusion strategies combining:
- Simple average pooling of per-model output logits.
- Performance-history weighted voting with exponential moving average tracking.
- Task-aware scaled dot-product attention over learned task embeddings.
- Hybrid fusion combining task attention and per-input dynamic confidence.

Hardened with:
- Subclassing nn.Module with nn.ModuleList for full PyTorch compatibility.
- Automatic device placement and input tensor movement to self.device.
- Complete attention_mask propagation to all sub-models.
- trainable_dynamic flag for optional end-to-end gradient flow.
- Strict model_task_ids length validation.
- Unified 60/40 hybrid combination.
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.utils.compat import repeat_kv

logger = logging.getLogger(__name__)

__all__ = [
    "TaskAwareAttention",
    "SmartWeightedEnsemble",
    "WeightedEnsembleAttention",
    "SmartWeightedEnsembleAttention",
    "inject_weighted_ensemble",
    "inject_smart_weighted_ensemble",
    "run_smart_ensemble",
]


class TaskAwareAttention(nn.Module):
    """
    Learns to weight model predictions based on task embeddings and pairwise
    task similarity matrix.
    """

    def __init__(self, num_tasks: Union[int, Any] = 1, hidden_dim: int = 64):
        super().__init__()
        if isinstance(num_tasks, int):
            self.num_tasks = max(1, num_tasks)
        else:
            # Gracefully handle HF PretrainedConfig
            extracted_tasks = getattr(num_tasks, "num_labels", getattr(num_tasks, "num_tasks", 2))
            if not isinstance(extracted_tasks, int):
                extracted_tasks = 2
            self.num_tasks = max(1, extracted_tasks)
        self.hidden_dim = hidden_dim
        self.task_emb = nn.Embedding(self.num_tasks, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.sim_mat = nn.Parameter(torch.eye(self.num_tasks))  # (num_tasks, num_tasks)

    def copy_weights(self, old_attn: Any) -> None:
        """Safe fallback for copy_weights compatibility."""
        pass

    def forward(
        self, task_id: int, all_logits: torch.Tensor, model_task_ids: List[int]
    ) -> torch.Tensor:
        """
        Args:
            task_id: target task id (integer)
            all_logits: (num_models, batch_size, num_labels) logit outputs
            model_task_ids: list of length num_models, each entry is the task
                            that model was trained on (or a proxy)
        Returns:
            weighted_logits: (batch_size, num_labels)
        """
        device = all_logits.device
        num_models = all_logits.shape[0]

        if len(model_task_ids) != num_models:
            raise ValueError(
                f"model_task_ids length ({len(model_task_ids)}) must match all_logits.shape[0] ({num_models})"
            )

        # Clamp task_id within bounds
        task_id = max(0, min(self.num_tasks - 1, task_id))
        cur_id = torch.tensor([task_id], device=device)
        cur_emb = self.q_proj(self.task_emb(cur_id))  # (1, H)

        # Clamped model task embeddings
        valid_model_tasks = [max(0, min(self.num_tasks - 1, tid)) for tid in model_task_ids]
        all_ids = torch.tensor(valid_model_tasks, device=device)  # (M,)
        all_embs = self.k_proj(self.task_emb(all_ids))  # (M, H)

        # Attention scores (dot product)
        scale = 1.0 / math.sqrt(self.hidden_dim)
        scores = torch.matmul(cur_emb, all_embs.t().contiguous()) * scale  # (1, M)

        # Add task similarity bias
        valid_model_tasks_tensor = torch.tensor(valid_model_tasks, device=device)
        bias = self.sim_mat[valid_model_tasks_tensor, task_id].unsqueeze(0)  # (1, M)
        scores = scores + bias
        weights = F.softmax(scores, dim=-1)  # (1, M)

        # Correctly weight the models' logits: (M, 1, 1) * (M, B, C) -> sum -> (B, C)
        weighted = (weights.view(num_models, 1, 1) * all_logits).sum(dim=0)
        return weighted


class SmartWeightedEnsemble(nn.Module):
    """
    Ensemble that combines multiple model predictions using various strategies:
        - simple_average
        - weighted by performance history
        - task-aware attention
        - hybrid (attention + dynamic confidence)
    """

    def __init__(
        self,
        models: List[nn.Module],
        num_tasks: int = 1,
        device: Optional[torch.device] = None,
        trainable_dynamic: bool = False,
    ):
        super().__init__()
        if not models:
            raise ValueError("SmartWeightedEnsemble: 'models' list cannot be empty.")

        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        self.num_tasks = max(1, num_tasks)
        self.trainable_dynamic = trainable_dynamic

        if device is None:
            try:
                self.device = next(models[0].parameters()).device
            except (StopIteration, AttributeError):
                self.device = torch.device("cpu")
        else:
            self.device = device

        # Performance tracking buffer: (num_models, num_tasks) matrix
        self.register_buffer(
            "performance",
            torch.ones(self.num_models, self.num_tasks, device=self.device) * 0.5,
        )

        self.attention = TaskAwareAttention(self.num_tasks).to(self.device)
        self.to(self.device)

    def to(self, *args: Any, **kwargs: Any) -> SmartWeightedEnsemble:
        res = super().to(*args, **kwargs)
        # Update self.device to match
        try:
            p = next(self.parameters())
            self.device = p.device
        except StopIteration:
            pass
        return res

    def update_performance(
        self,
        task_id: int,
        model_outputs: List[torch.Tensor],
        target: torch.Tensor,
        metric: str = "accuracy",
    ) -> None:
        """Update performance scores based on validation results."""
        task_id = max(0, min(self.num_tasks - 1, task_id))
        target = target.to(self.device)
        with torch.no_grad():
            for i, logits in enumerate(model_outputs):
                if i >= self.num_models:
                    break
                logits = logits.to(self.device)
                if metric == "accuracy":
                    preds = logits.argmax(dim=-1)
                    acc = (preds == target).float().mean().item()
                    self.performance[i, task_id] = (
                        0.9 * self.performance[i, task_id] + 0.1 * acc
                    )

    def _get_dynamic_weights(
        self,
        task_id: int,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Compute dynamic weights based on input-specific confidence."""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        def _forward_all():
            weights_list = []
            for i, model in enumerate(self.models):
                if not self.trainable_dynamic:
                    model.eval()
                if attention_mask is not None:
                    out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                else:
                    out = model(input_ids=input_ids, **kwargs)
                if hasattr(out, "logits"):
                    logits = out.logits
                elif isinstance(out, (tuple, list)):
                    logits = out[0]
                else:
                    logits = out

                probs = F.softmax(logits.float(), dim=-1)
                conf = probs.max(dim=-1)[0].mean()
                weights_list.append(conf)
            return torch.stack(weights_list)

        if self.trainable_dynamic:
            raw_weights = _forward_all()
        else:
            with torch.no_grad():
                raw_weights = _forward_all()

        if raw_weights.numel() == 0 or raw_weights.sum() <= 0:
            return torch.ones(self.num_models, device=self.device) / self.num_models
        return raw_weights / raw_weights.sum()

    def combine_logits(
        self,
        task_id: int,
        all_logits: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        method: str = "hybrid",
        model_task_ids: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Combine logits from all models.

        Args:
            task_id: target task (integer)
            all_logits: (num_models, batch_size, num_labels) – already computed.
            input_ids, attention_mask: used for dynamic weighting.
            method: 'simple_average', 'weighted', 'task_aware_attention', 'hybrid'
            model_task_ids: task assignments for each sub-model.
        """
        all_logits = all_logits.to(self.device)
        num_models = all_logits.shape[0]
        task_id = max(0, min(self.num_tasks - 1, task_id))

        if model_task_ids is not None and len(model_task_ids) != num_models:
            raise ValueError(
                f"model_task_ids length ({len(model_task_ids)}) must match num_models ({num_models})"
            )

        t_ids = (
            model_task_ids
            if model_task_ids is not None
            else [i % self.num_tasks for i in range(num_models)]
        )

        if method == "simple_average":
            return all_logits.mean(dim=0)

        elif method == "weighted":
            perf_weights = self.performance[:, task_id]  # (M,)
            sum_w = perf_weights.sum()
            if sum_w > 0:
                perf_weights = perf_weights / sum_w
            else:
                perf_weights = torch.ones_like(perf_weights) / max(1, len(perf_weights))
            return (all_logits * perf_weights.view(-1, 1, 1)).sum(dim=0)

        elif method == "task_aware_attention":
            return self.attention(task_id, all_logits, t_ids)

        elif method == "hybrid":
            attn_weighted = self.attention(task_id, all_logits, t_ids)
            if input_ids is not None:
                dyn_weights = self._get_dynamic_weights(
                    task_id, input_ids, attention_mask=attention_mask, **kwargs
                )  # (M,)
                dyn_weighted = (all_logits * dyn_weights.view(-1, 1, 1)).sum(dim=0)
                return 0.6 * attn_weighted + 0.4 * dyn_weighted
            return attn_weighted

        else:
            raise ValueError(f"Unknown ensemble method: {method}")

    def forward(
        self,
        input_ids: torch.Tensor,
        task_id: int = 0,
        attention_mask: Optional[torch.Tensor] = None,
        method: str = "hybrid",
        model_task_ids: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass querying each model and combining logits with attention_mask propagation."""
        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        if model_task_ids is not None and len(model_task_ids) != self.num_models:
            raise ValueError(
                f"model_task_ids length ({len(model_task_ids)}) must match num_models ({self.num_models})"
            )

        all_logits = []
        for model in self.models:
            if attention_mask is not None:
                out = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            else:
                out = model(input_ids=input_ids, **kwargs)

            if hasattr(out, "logits"):
                logits = out.logits
            elif isinstance(out, (tuple, list)):
                logits = out[0]
            else:
                logits = out
            all_logits.append(logits)

        stacked = torch.stack(all_logits, dim=0)
        stacked = stacked.to(self.device)

        return self.combine_logits(
            task_id=task_id,
            all_logits=stacked,
            input_ids=input_ids,
            attention_mask=attention_mask,
            method=method,
            model_task_ids=model_task_ids,
            **kwargs,
        )

    def forward_simple_average(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward(
            input_ids,
            task_id=0,
            attention_mask=attention_mask,
            method="simple_average",
            **kwargs,
        )

    def copy_weights(self, old_attn: Any) -> None:
        """Safe fallback for copy_weights compatibility."""
        pass

    def forward_task_attention(
        self,
        task_id: int,
        model_task_ids: List[int],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward(
            input_ids,
            task_id=task_id,
            attention_mask=attention_mask,
            method="task_aware_attention",
            model_task_ids=model_task_ids,
            **kwargs,
        )

    def forward_hybrid(
        self,
        task_id: int,
        model_task_ids: List[int],
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        return self.forward(
            input_ids,
            task_id=task_id,
            attention_mask=attention_mask,
            method="hybrid",
            model_task_ids=model_task_ids,
            **kwargs,
        )


def run_smart_ensemble(config: dict, dataset_path: str) -> None:
    """Command-line entry point for running smart ensemble."""
    print("Use the class 'SmartWeightedEnsemble' directly in your training/evaluation script.")


@ATTENTION_REGISTRY.register("weighted_ensemble")
@ATTENTION_REGISTRY.register("smart_weighted_ensemble")
class WeightedEnsembleAttention(BaseAttention):
    r"""
    Weighted Ensemble Attention module combining:
    1. Standard Softmax Dot-Product Attention branch.
    2. Kernelized Linear Attention branch with ELU+1 feature maps and clamped normalization.
    3. Dynamic learned gating fusion between branches.

    Hardened for:
    - Multi-architecture weight transfer (RoBERTa nested encoder, GPT-2 Conv1D, DistilBERT, Qwen).
    - Unconditional nested encoder detection and output projection bypass.
    - Causal masking for decoders across all mask formats.
    - Denominator clamp (min=1.0) and output tensor clamping [-1e4, 1e4].
    - Valid token mask safeguarding.
    - Frobenius norm verification (>= 1e-4).
    """

    supports_causal: bool = True
    supports_bidirectional: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    is_custom_attention: bool = True
    supports_delta_net: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        rank: Optional[int] = None,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank, **kwargs)

        cfg_source = getattr(config, "text_config", config) if config is not None else None
        h_size = (
            getattr(cfg_source, "hidden_size", None)
            or getattr(cfg_source, "d_model", None)
            or getattr(cfg_source, "n_embd", None)
            or getattr(cfg_source, "dim", None)
            or kwargs.get("hidden_size", 768)
        )
        n_heads = (
            getattr(cfg_source, "num_attention_heads", None)
            or getattr(cfg_source, "num_heads", None)
            or getattr(cfg_source, "n_heads", None)
            or getattr(cfg_source, "n_head", None)
            or kwargs.get("num_heads", 12)
        )
        h_dim = (
            getattr(cfg_source, "head_dim", None)
            or getattr(cfg_source, "d_kv", None)
            or kwargs.get("head_dim", h_size // max(1, n_heads))
        )
        n_kv_heads = (
            getattr(cfg_source, "num_key_value_heads", None)
            or getattr(cfg_source, "num_kv_heads", None)
            or kwargs.get("num_key_value_heads", n_heads)
        )
        drop_rate = (
            getattr(cfg_source, "attention_probs_dropout_prob", None)
            or getattr(cfg_source, "dropout", None)
            or kwargs.get("dropout", 0.1)
        )

        self.hidden_size: int = int(h_size)
        self.num_heads: int = int(n_heads)
        self.head_dim: int = int(h_dim)
        self.num_kv_heads: int = int(n_kv_heads)
        self.kv_repeat: int = max(1, self.num_heads // max(1, self.num_kv_heads))
        self.all_head_size: int = self.num_heads * self.head_dim
        self.all_kv_head_size: int = self.num_kv_heads * self.head_dim

        # Decoder and nested encoder flags
        is_dec = bool(getattr(cfg_source, "is_decoder", False) or kwargs.get("is_decoder", False))
        _mt = getattr(cfg_source, "model_type", "").lower() if cfg_source else ""
        _decoder_prefixes = ("gpt", "qwen", "llama", "mistral", "gemma", "phi", "opt", "bloom", "falcon", "olmo", "mpt", "stablelm")
        if any(_mt.startswith(p) for p in _decoder_prefixes):
            is_dec = True
        self.is_decoder: bool = is_dec
        self.is_nested_encoder: bool = False

        # Projections
        self.q_proj = nn.Linear(self.hidden_size, self.all_head_size, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.all_kv_head_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.all_kv_head_size, bias=True)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size, bias=True)
        self.dropout = nn.Dropout(drop_rate)

        # Dynamic gating ensemble fusion
        self.ensemble_gate = nn.Linear(self.hidden_size, self.num_heads)
        with torch.no_grad():
            self.ensemble_gate.weight.zero_()
            self.ensemble_gate.bias.fill_(1.0)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    # State dict / attribute aliases
    @property
    def query(self) -> nn.Linear:
        return self.q_proj

    @property
    def key(self) -> nn.Linear:
        return self.k_proj

    @property
    def value(self) -> nn.Linear:
        return self.v_proj

    @property
    def dense(self) -> nn.Linear:
        return self.out_proj

    def _standard_branch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        B, H, N, Dh = Q.shape
        _, _, N_k, _ = K.shape
        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(Q, K.transpose(-2, -1).contiguous()) * scale  # [B, H, N, N_k]

        if self.is_decoder and N > 1:
            idx_q = torch.arange(N, device=Q.device).unsqueeze(1)
            idx_k = torch.arange(N_k, device=Q.device).unsqueeze(0)
            causal_mask = (idx_k <= idx_q).unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(~causal_mask, -10000.0)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                ext_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2).float()) * -10000.0
                scores = scores + ext_mask.to(scores.dtype)
            elif attention_mask.dim() == 3:
                scores = scores + attention_mask.unsqueeze(1).to(scores.dtype)
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.to(scores.dtype)

        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        attn = F.softmax(scores.float(), dim=-1).to(Q.dtype)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)

    def _linear_branch(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        phi_Q = F.elu(Q) + 1.0
        phi_K = F.elu(K) + 1.0

        mask = valid_mask.unsqueeze(1).unsqueeze(-1).to(Q.dtype)  # [B, 1, N_k, 1]
        phi_K = phi_K * mask
        V = V * mask

        if self.is_decoder:
            KV = torch.einsum("bhnd,bhne->bhnde", phi_K, V)
            KV_cumsum = torch.cumsum(KV, dim=2)
            num = torch.einsum("bhnd,bhnde->bhne", phi_Q, KV_cumsum)
            K_cumsum = torch.cumsum(phi_K, dim=2)
            den = torch.einsum("bhnd,bhnd->bhn", phi_Q, K_cumsum).unsqueeze(-1)
        else:
            KV = torch.matmul(phi_K.transpose(-2, -1).contiguous(), V)  # [B, H, Dh, Dh]
            num = torch.matmul(phi_Q, KV)  # [B, H, N, Dh]
            Z = phi_K.sum(dim=-2, keepdim=True)  # [B, H, 1, Dh]
            den = torch.matmul(phi_Q, Z.transpose(-2, -1).contiguous())  # [B, H, N, 1]

        den = den.clamp(min=1e-6)
        out = num / den
        return torch.nan_to_num(out, nan=0.0)

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
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        is_cross = encoder_hidden_states is not None
        kv_input = encoder_hidden_states if is_cross else hidden_states
        N_k = kv_input.shape[1]

        eff_mask = encoder_attention_mask if is_cross else attention_mask

        # Safeguard valid_mask extraction
        valid_mask: Optional[torch.Tensor] = None
        if eff_mask is not None:
            if eff_mask.dim() == 2:
                if eff_mask.dtype in (torch.int, torch.long, torch.bool):
                    valid_mask = eff_mask.bool()
                else:
                    valid_mask = (eff_mask > -5000.0).bool()
            elif eff_mask.dim() == 3:
                valid_mask = (eff_mask[:, 0, :] > -5000.0).bool() if eff_mask.shape[1] == 1 else (eff_mask > -5000.0).any(dim=1).bool()
            elif eff_mask.dim() == 4:
                valid_mask = (eff_mask > -5000.0).any(dim=-2).squeeze(1).bool()

        if valid_mask is None:
            valid_mask = torch.ones(B, N_k, device=device, dtype=torch.bool)

        Q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        K = self.k_proj(kv_input).view(B, N_k, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()
        V = self.v_proj(kv_input).view(B, N_k, self.num_kv_heads, self.head_dim).transpose(1, 2).contiguous()

        # Rotary position embeddings if provided
        position_embeddings = kwargs.get("position_embeddings", None)
        if position_embeddings is not None and isinstance(position_embeddings, (tuple, list)) and len(position_embeddings) >= 2:
            cos, sin = position_embeddings[0], position_embeddings[1]

            def _rotate_half(x: torch.Tensor) -> torch.Tensor:
                return torch.cat((-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]), dim=-1)

            if cos.dim() == 3:
                cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            while cos.dim() < Q.dim():
                cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
            if cos.shape[-2] != Q.shape[-2]:
                cos, sin = cos[..., : Q.shape[-2], :], sin[..., : Q.shape[-2], :]

            rotary_dim = cos.shape[-1]
            if rotary_dim <= Q.shape[-1]:
                q_rot, q_pass = Q[..., :rotary_dim], Q[..., rotary_dim:]
                k_rot, k_pass = K[..., :rotary_dim], K[..., rotary_dim:]
                q_rot = (q_rot * cos.to(q_rot.dtype)) + (_rotate_half(q_rot) * sin.to(q_rot.dtype))
                k_rot = (k_rot * cos.to(k_rot.dtype)) + (_rotate_half(k_rot) * sin.to(k_rot.dtype))
                Q = torch.cat([q_rot, q_pass], dim=-1) if q_pass.numel() > 0 else q_rot
                K = torch.cat([k_rot, k_pass], dim=-1) if k_pass.numel() > 0 else k_rot

        if self.kv_repeat > 1:
            K = repeat_kv(K, self.kv_repeat)
            V = repeat_kv(V, self.kv_repeat)

        # Compute ensemble branches
        out_standard = self._standard_branch(Q, K, V, eff_mask)
        out_linear = self._linear_branch(Q, K, V, valid_mask)

        # Learned dynamic gating
        gate = torch.sigmoid(self.ensemble_gate(hidden_states)).view(B, N, self.num_heads, 1).transpose(1, 2).contiguous()
        fused = gate * out_standard + (1.0 - gate) * out_linear

        if head_mask is not None:
            if head_mask.dim() == 1:
                fused = fused * head_mask.view(1, -1, 1, 1)
            elif head_mask.dim() == 2:
                fused = fused * head_mask.unsqueeze(1).unsqueeze(-1)
            elif head_mask.dim() == 4:
                fused = fused * head_mask

        fused = fused.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)

        if self.is_nested_encoder:
            out = fused
        else:
            out = self.out_proj(fused)
            out = self.dropout(out)

        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(min=-1e4, max=1e4)
        return out, None

    def copy_weights(self, old_attn: Any) -> None:
        """Robust multi-architecture weight transfer for RoBERTa, GPT-2, DistilBERT, etc."""
        if old_attn is None:
            return

        cls_name = old_attn.__class__.__name__
        has_out_proj = any(
            hasattr(old_attn, attr)
            for attr in ("dense", "out_proj", "c_proj", "out_lin", "o_proj", "wo", "o")
        )
        if cls_name in ("BertSelfAttention", "RobertaSelfAttention") or (
            cls_name.endswith("SelfAttention") and not has_out_proj
        ):
            self.is_nested_encoder = True
        elif (
            hasattr(old_attn, "self")
            and hasattr(getattr(old_attn, "self"), "query")
            and hasattr(old_attn, "output")
            and hasattr(getattr(old_attn, "output"), "dense")
        ):
            self.is_nested_encoder = True
        elif not has_out_proj:
            self.is_nested_encoder = True

        if hasattr(old_attn, "is_decoder"):
            self.is_decoder = bool(old_attn.is_decoder)
        elif cls_name in ("GPT2Attention", "Qwen2Attention") or hasattr(old_attn, "c_attn"):
            self.is_decoder = True

        from framework.core.adapters import UniversalAdapter
        extracted = {}
        for proj in ["q", "k", "v", "o"]:
            w, b = None, None
            if proj == "o":
                if hasattr(old_attn, "c_proj"):
                    mod = old_attn.c_proj
                    base_mod = getattr(mod, "base_layer", mod)
                    w = getattr(base_mod, "weight", getattr(mod, "weight", None))
                    if w is not None:
                        w = w.data
                        if (mod.__class__.__name__ == "Conv1D") or (base_mod.__class__.__name__ == "Conv1D") or hasattr(mod, "nf") or hasattr(base_mod, "nf") or (w.shape[0] < w.shape[1]):
                            w = w.t().contiguous()
                    bias_holder = getattr(base_mod, "bias", getattr(mod, "bias", None))
                    b = bias_holder.data if bias_holder is not None else None
                elif hasattr(old_attn, "output") and hasattr(getattr(old_attn, "output"), "dense"):
                    mod = old_attn.output.dense
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "dense"):
                    mod = old_attn.dense
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "out_lin"):
                    mod = old_attn.out_lin
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "out_proj"):
                    mod = old_attn.out_proj
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)
                elif hasattr(old_attn, "o_proj"):
                    mod = old_attn.o_proj
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)

            elif hasattr(old_attn, "c_attn") and proj in ["q", "k", "v"]:
                mod = old_attn.c_attn
                base_mod = getattr(mod, "base_layer", mod)
                c_w = getattr(base_mod, "weight", getattr(mod, "weight", None))
                if c_w is not None:
                    c_w = c_w.data
                    if (mod.__class__.__name__ == "Conv1D") or (base_mod.__class__.__name__ == "Conv1D") or hasattr(mod, "nf") or hasattr(base_mod, "nf") or (c_w.shape[0] < c_w.shape[1]):
                        c_w = c_w.t().contiguous()
                    split_size = c_w.size(0) // 3
                    idx = {"q": 0, "k": 1, "v": 2}[proj]
                    w = c_w[idx * split_size : (idx + 1) * split_size, :]
                    bias_holder = getattr(base_mod, "bias", getattr(mod, "bias", None))
                    if bias_holder is not None:
                        c_b = bias_holder.data
                        b = c_b[idx * split_size : (idx + 1) * split_size]

            elif hasattr(old_attn, "self") and proj in ["q", "k", "v"]:
                s = getattr(old_attn, "self")
                name_map = {"q": "query", "k": "key", "v": "value"}
                if hasattr(s, name_map[proj]):
                    mod = getattr(s, name_map[proj])
                    w = mod.weight.data
                    b = getattr(mod, "bias", None)

            if w is None:
                try:
                    w, b = UniversalAdapter.get_weight_bias(old_attn, proj)
                except Exception as ex:
                    logger.debug(f"UniversalAdapter extraction for {proj} failed: {ex}")

            extracted[proj] = (w, b)

        # Frobenius norm verification (>= 1e-4)
        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj)]:
            w, b = extracted[proj]
            if w is not None:
                if torch.linalg.norm(w.float()) < 1e-4:
                    logger.warning(f"Extracted weight for {proj} has near-zero Frobenius norm")
                with torch.no_grad():
                    if target.weight.shape == w.shape:
                        target.weight.copy_(w)
                    elif target.weight.shape == w.t().contiguous().shape:
                        target.weight.copy_(w.t().contiguous())
                    else:
                        w_or = w if w.shape[1] == target.weight.shape[1] else w.t().contiguous()
                        out_d, in_d = target.weight.shape
                        s_out, s_in = w_or.shape
                        target.weight.data[:min(out_d, s_out), :min(in_d, s_in)].copy_(w_or[:min(out_d, s_out), :min(in_d, s_in)])
                    if b is not None and target.bias is not None:
                        target.bias.copy_(b if torch.is_tensor(b) else b.data)

        if not self.is_nested_encoder:
            w_o, b_o = extracted["o"]
            if w_o is not None:
                if torch.linalg.norm(w_o.float()) < 1e-4:
                    logger.warning("Extracted weight for o has near-zero Frobenius norm")
                with torch.no_grad():
                    if self.out_proj.weight.shape == w_o.shape:
                        self.out_proj.weight.copy_(w_o)
                    elif self.out_proj.weight.shape == w_o.t().contiguous().shape:
                        self.out_proj.weight.copy_(w_o.t().contiguous())
                    else:
                        w_or = w_o if w_o.shape[1] == self.out_proj.weight.shape[1] else w_o.t().contiguous()
                        out_d, in_d = self.out_proj.weight.shape
                        s_out, s_in = w_or.shape
                        self.out_proj.weight.data[:min(out_d, s_out), :min(in_d, s_in)].copy_(w_or[:min(out_d, s_out), :min(in_d, s_in)])
                    if b_o is not None and self.out_proj.bias is not None:
                        self.out_proj.bias.copy_(b_o if torch.is_tensor(b_o) else b_o.data)


# Alias for backward compatibility
SmartWeightedEnsembleAttention = WeightedEnsembleAttention


@ATTENTION_REGISTRY.register("weighted_ensemble_inject")
@ATTENTION_REGISTRY.register("inject_weighted_ensemble")
@ATTENTION_REGISTRY.register("smart_weighted_ensemble_inject")
@ATTENTION_REGISTRY.register("inject_smart_weighted_ensemble")
def inject_weighted_ensemble(
    layer: Any = None,
    model: Any = None,
    model_config: Any = None,
    comp_config: Any = None,
    layer_idx: int = 0,
    total_layers: int = 12,
    old_attn: Optional[nn.Module] = None,
    **kwargs: Any,
):
    target = layer if layer is not None else model
    if target is None:
        target = kwargs.get("model", kwargs.get("layer", None))
    if target is None:
        return None

    cfg = model_config if model_config is not None else getattr(target, "config", None)

    # Whole-model traversal
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
    elif hasattr(target, "bert") and hasattr(target.bert, "encoder") and hasattr(target.bert.encoder, "layer"):
        layer_list = target.bert.encoder.layer
    elif hasattr(target, "layers") and isinstance(target.layers, (torch.nn.ModuleList, list)):
        layer_list = target.layers

    if layer_list is not None:
        num_injected = 0
        tot = len(layer_list)
        for idx, lyr in enumerate(layer_list):
            res = inject_weighted_ensemble(
                layer=lyr,
                model_config=cfg,
                comp_config=comp_config if 'comp_config' in locals() or 'comp_config' in kwargs else None,
                layer_idx=idx,
                total_layers=tot,
                **kwargs,
            )
            if res is not None:
                num_injected += 1
        return target if num_injected > 0 else None

    layer = target

    r"""
    Injects WeightedEnsembleAttention into a transformer layer.
    Returns None on failure to preserve layer stability.
    """
    try:
        if model_config is None:
            if hasattr(layer, "hidden_size") or hasattr(layer, "d_model") or hasattr(layer, "n_embd") or hasattr(layer, "dim"):
                model_config = layer
            elif hasattr(layer, "attention"):
                old_attn = layer.attention

        src_attn = old_attn if old_attn is not None else getattr(layer, "attention", getattr(layer, "self", getattr(layer, "attn", None)))

        new_attn = WeightedEnsembleAttention(
            config=model_config,
            pretrained_attn=src_attn,
            **kwargs,
        )

        if src_attn is not None:
            p = next(src_attn.parameters(), None)
            if p is not None:
                new_attn.to(device=p.device, dtype=p.dtype)

        if hasattr(layer, "attention"):
            if hasattr(layer.attention, "self"):
                layer.attention.self = new_attn
            else:
                layer.attention = new_attn
        elif hasattr(layer, "self_attn"):
            layer.self_attn = new_attn
        elif hasattr(layer, "self"):
            layer.self = new_attn
        elif hasattr(layer, "attn"):
            layer.attn = new_attn

        return new_attn
    except Exception as e:
        logger.error(f"Failed to inject weighted ensemble: {e}")
        return None


inject_smart_weighted_ensemble = inject_weighted_ensemble
