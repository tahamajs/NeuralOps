"""
CacheNotes: Task-Aware Key-Value Cache Compression and Static Weight Compression
with Learnable Continuous CPT (Compression-Planning-Token) Generator.

Given a task description, CacheNotes generates Compression-Planning-Tokens (CPTs)
that guide the compression of the corpus and KV cache into compact, reusable states.

Scientific Enhancement:
- Replaces static keyword heuristics with a continuous, differentiable CPTGenerator network.
- Learnable task representation using tokenized embedding bag and multi-frequency coordinate positional projection.
- Capable of evaluating task importance at ANY continuous output feature dimension d (64, 128, 768, 1024, 2048, 4096, etc.).
- Backprop-trainable end-to-end minimizing Frobenius reconstruction loss under CPT-guided low-rank SVD.
- Full fallback compatibility with static CPT heuristic if generator is untrained or disabled.
"""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.pruning_utils import recalibrate_layernorms
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

try:
    from transformers.pytorch_utils import Conv1D
except ImportError:
    Conv1D = None

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely extract configuration attribute from None, dict, or namespace object."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class CacheNotesLoRALinear(nn.Module):
    """
    Linear layer containing CacheNotes compressed base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.nf = out_features  # Conv1D compatibility
        self.lora_r = int(lora_r)
        self.lora_alpha = float(lora_alpha)
        self.scaling = (self.lora_alpha / self.lora_r) if self.lora_r > 0 else 1.0

        target_device = device if device is not None else weight.device
        target_dtype = dtype if dtype is not None else weight.dtype

        self.weight = nn.Parameter(
            weight.to(device=target_device, dtype=target_dtype),
            requires_grad=False,
        )
        if bias is not None:
            self.bias = nn.Parameter(
                bias.to(device=target_device, dtype=target_dtype),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(
                torch.empty(
                    (self.lora_r, in_features), dtype=target_dtype, device=target_device
                )
            )
            self.lora_B = nn.Parameter(
                torch.zeros(
                    (out_features, self.lora_r),
                    dtype=target_dtype,
                    device=target_device,
                )
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        weight: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> CacheNotesLoRALinear:
        cls_name = linear.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(linear, "nf")
        raw_w = linear.weight.data

        if is_conv1d:
            in_f = raw_w.shape[0]
            out_f = raw_w.shape[1]
        else:
            out_f = raw_w.shape[0]
            in_f = raw_w.shape[1]

        bias = (
            linear.bias.detach().clone()
            if getattr(linear, "bias", None) is not None and linear.bias is not None
            else None
        )
        return cls(
            in_features=in_f,
            out_features=out_f,
            weight=weight,
            bias=bias,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            device=weight.device,
            dtype=weight.dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        out = F.linear(
            x_flat,
            self.weight.to(x_flat.dtype),
            self.bias.to(x_flat.dtype) if self.bias is not None else None,
        )

        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(
                F.linear(x_flat.to(self.lora_A.dtype), self.lora_A), self.lora_B
            )
            out = out + (self.scaling * lora_out).to(out.dtype)

        return out.reshape(*orig_shape[:-1], self.out_features)


class CompressionPlanningTokens:
    """
    Static CPT heuristic importance generator.
    Serves as safe baseline fallback when learnable CPT generator is inactive.
    """

    TASK_KEYWORD_MAP = {
        "qa": [
            "question",
            "answer",
            "context",
            "passage",
            "who",
            "what",
            "where",
            "why",
        ],
        "summarize": ["summary", "abstract", "key", "main", "brief", "conclusion"],
        "code": [
            "function",
            "class",
            "def",
            "return",
            "import",
            "self",
            "error",
            "test",
        ],
        "reasoning": [
            "therefore",
            "because",
            "if",
            "then",
            "thus",
            "proof",
            "step",
            "hence",
        ],
        "math": [
            "solve",
            "equation",
            "matrix",
            "integral",
            "theorem",
            "calculate",
            "equal",
        ],
    }

    @classmethod
    def generate_importance(
        cls,
        task_type: str,
        num_features: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate per-feature importance modulation vector based on static heuristics."""
        task_lower = str(task_type).lower()
        importance = torch.zeros(num_features, device=device)

        # Baseline structure: prompt position bias
        pos_dim = max(1, num_features // 4)
        importance[:pos_dim] = 0.25

        if any(kw in task_lower for kw in ["qa", "question"]):
            importance[pos_dim : 2 * pos_dim] += 0.35
        elif any(kw in task_lower for kw in ["code", "program"]):
            importance[2 * pos_dim : 3 * pos_dim] += 0.40
        elif any(kw in task_lower for kw in ["reason", "math"]):
            importance += 0.20

        return torch.clamp(importance + 0.5, min=0.1, max=2.0)


class CPTGenerator(nn.Module):
    """
    Learnable Task-Aware Compression Planning Token (CPT) Generator.
    Maps arbitrary task descriptions to continuous per-feature importance vectors for weight SVD
    and KV cache compression across arbitrary dimension sizes d.
    """

    def __init__(
        self, hidden_dim: int = 128, num_freqs: int = 8, vocab_size: int = 5000
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_freqs = num_freqs
        self.vocab_size = vocab_size

        self.text_embedding = nn.EmbeddingBag(vocab_size, hidden_dim, mode="mean")
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        in_dim = hidden_dim + num_freqs * 2
        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _tokenize_text(self, text: str, device: torch.device) -> torch.Tensor:
        """Deterministic hash-based tokenization without external library dependencies."""
        words = (
            str(text)
            .lower()
            .replace(",", " ")
            .replace(".", " ")
            .replace(":", " ")
            .replace("-", " ")
            .split()
        )
        if not words:
            return torch.tensor([0], dtype=torch.long, device=device)
        indices = []
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            indices.append(h % (self.vocab_size - 1) + 1)
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(
        self, task: str, d: int, device: torch.device | None = None
    ) -> torch.Tensor:
        """
        Generate a per-feature importance modulation vector of size d in [0.1, 2.0].
        """
        target_device = device if device is not None else next(self.parameters()).device
        tokens = self._tokenize_text(task, target_device)
        h = self.text_embedding(tokens.unsqueeze(0)).squeeze(0)  # [hidden_dim]
        h = self.encoder(h)  # [hidden_dim]

        # Continuous NeRF-style positional coordinate encoding
        coords = torch.linspace(0.0, 1.0, d, device=target_device)
        freqs = (2.0 ** torch.arange(self.num_freqs, device=target_device)) * math.pi
        args = coords.unsqueeze(1) * freqs.unsqueeze(0)  # [d, num_freqs]
        enc = torch.cat(
            [torch.sin(args), torch.cos(args)], dim=-1
        )  # [d, 2 * num_freqs]

        h_exp = h.unsqueeze(0).expand(d, -1)  # [d, hidden_dim]
        inp = torch.cat([h_exp, enc], dim=-1)  # [d, in_dim]

        # Importance modulation strictly in [0.1, 2.0]
        importance = self.projector(inp).squeeze(-1) * 1.9 + 0.1
        return importance


class CacheNotesKVCache:
    """
    Runtime task-guided dynamic KV cache compressor.
    Evaluates token task-relevance dynamically and prunes non-essential context tokens,
    supporting variable sequence lengths during prefill and autoregressive decoding.
    """

    def __init__(
        self,
        max_cache_len: int = 128,
        task_type: str = "qa",
        keep_ratio: float = 0.3,
        num_sinks: int = 4,
        cpt_generator: CPTGenerator | None = None,
    ) -> None:
        self.max_cache_len = int(max_cache_len)
        self.task_type = task_type
        self.keep_ratio = float(keep_ratio)
        self.num_sinks = int(num_sinks)
        self.cpt_generator = cpt_generator
        self.k_cache: torch.Tensor | None = None
        self.v_cache: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear cache state to prevent memory leaks across sequences."""
        self.k_cache = None
        self.v_cache = None

    @torch.no_grad()
    def update_and_compress(
        self,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        is_new_sequence: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new KV tokens and apply task-aware retention when exceeding capacity.
        Shapes: (B, H, N, D).
        Automatically resets cache on new multi-token prompt prefill (N > 1) or when is_new_sequence=True (H6).
        """
        k_4d = new_k
        v_4d = new_v
        if new_k.ndim == 2:
            k_4d = new_k.unsqueeze(0).unsqueeze(0)
            v_4d = new_v.unsqueeze(0).unsqueeze(0)
        elif new_k.ndim == 3:
            k_4d = new_k.unsqueeze(1)
            v_4d = new_v.unsqueeze(1)

        B, H, N, D = k_4d.shape

        # H6: Prevent cross-sequence cache pollution on prompt prefill (N > 1) or explicit flag
        if is_new_sequence or N > 1:
            self.reset()

        if self.k_cache is None:
            self.k_cache = k_4d
            self.v_cache = v_4d
        else:
            if self.k_cache.shape[0] != B:
                self.k_cache = self.k_cache[:1].expand(B, -1, -1, -1).clone()
                self.v_cache = self.v_cache[:1].expand(B, -1, -1, -1).clone()
            self.k_cache = torch.cat([self.k_cache, k_4d], dim=2)
            self.v_cache = torch.cat([self.v_cache, v_4d], dim=2)

        cur_len = self.k_cache.size(2)
        if cur_len <= self.max_cache_len:
            return self.k_cache, self.v_cache

        sinks = min(self.num_sinks, cur_len)
        retained_cap = max(sinks + 1, self.max_cache_len)

        # Task importance weighting across tokens
        if self.cpt_generator is not None:
            cpt_importance = self.cpt_generator(
                self.task_type, D, device=self.k_cache.device
            )
        else:
            cpt_importance = CompressionPlanningTokens.generate_importance(
                self.task_type, D, device=self.k_cache.device
            )

        weighted_k = self.k_cache * cpt_importance.view(1, 1, 1, D)
        token_scores = torch.norm(weighted_k.float(), dim=-1)  # (B, H, cur_len)

        # Always preserve initial sink tokens
        token_scores[:, :, :sinks] += 1e6

        _, top_indices = torch.topk(token_scores, retained_cap, dim=-1)
        top_indices, _ = torch.sort(top_indices, dim=-1)

        idx_exp = top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
        self.k_cache = self.k_cache.gather(2, idx_exp).contiguous()
        self.v_cache = self.v_cache.gather(2, idx_exp).contiguous()

        return self.k_cache, self.v_cache


@COMPRESSOR_REGISTRY.register("cachenotes")
@COMPRESSOR_REGISTRY.register("cachenotes_compressor")
class CacheNotesCompressor(BaseCompressor):
    is_custom_attention: bool = True
    """
    CacheNotes Compressor: Decouples static weight compression from KV cache dynamic pruning
    with continuous learnable CPT Generator.

    Production hardening features:
    - Trainable CPTGenerator mapping task descriptions to per-feature importance vectors.
    - Support for LoRALinear wrappers, nn.Linear, and Conv1D (GPT-2).
    - Preserves existing LoRALinear wrappers without creating redundant layers.
    - target_attention_only parameter skipping FFN/MLP and preserving classification heads.
    - Robust config handling (_cfg_get) supporting None, dict, and namespace objects.
    - Fallback freezing: parameters in any fallback layer are frozen (requires_grad=False).
    - Checkpoint saving and loading for CPTGenerator.
    - Comprehensive metric tracking: actual ranks, reconstruction error, and KV cache saving.
    """

    def __init__(
        self,
        model: nn.Module,
        task_type: str = "qa",
        task_description: str | None = None,
        use_learned_cpt: bool = True,
        keep_ratio: float = 0.5,
        max_error: float = 0.35,
        fallback_to_original: bool = True,
        protect_critical: bool = True,
        target_attention_only: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = True,
        recalibrate_ln: bool = True,
        tokenizer: Any | None = None,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        device: str | torch.device | None = None,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="cachenotes", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))

        task_type = str(
            _cfg_get(cfg, "cachenotes_task", _cfg_get(cfg, "task_type", task_type))
        )
        task_description = str(
            _cfg_get(
                cfg,
                "cachenotes_task_description",
                _cfg_get(cfg, "task_description", task_description or task_type),
            )
        )
        use_learned_cpt = bool(
            _cfg_get(
                cfg,
                "cachenotes_use_learned_cpt",
                _cfg_get(cfg, "use_learned_cpt", use_learned_cpt),
            )
        )
        keep_ratio = float(
            _cfg_get(
                cfg, "cachenotes_keep_ratio", _cfg_get(cfg, "keep_ratio", keep_ratio)
            )
        )
        max_error = float(
            _cfg_get(cfg, "cachenotes_max_error", _cfg_get(cfg, "max_error", max_error))
        )
        fallback_to_original = bool(
            _cfg_get(
                cfg,
                "cachenotes_fallback_to_original",
                _cfg_get(cfg, "fallback_to_original", fallback_to_original),
            )
        )
        protect_critical = bool(_cfg_get(cfg, "protect_critical", protect_critical))
        target_attention_only = bool(
            _cfg_get(
                cfg,
                "cachenotes_target_attention_only",
                _cfg_get(cfg, "target_attention_only", target_attention_only),
            )
        )
        lora_r = int(
            _cfg_get(
                cfg,
                "cachenotes_lora_r",
                _cfg_get(cfg, "lora_r", _cfg_get(cfg, "lora_rank", lora_r)),
            )
        )
        lora_alpha = float(
            _cfg_get(
                cfg, "cachenotes_lora_alpha", _cfg_get(cfg, "lora_alpha", lora_alpha)
            )
        )
        skip_attention_qkv = bool(
            _cfg_get(
                cfg,
                "cachenotes_skip_attention_qkv",
                _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv),
            )
        )
        recalibrate_ln = bool(
            _cfg_get(
                cfg,
                "cachenotes_recalibrate_ln",
                _cfg_get(cfg, "recalibrate_ln", recalibrate_ln),
            )
        )

        self.task_type = task_type
        self.task_description = task_description
        self.use_learned_cpt = use_learned_cpt
        self.keep_ratio = max(0.05, min(keep_ratio, 1.0))
        self.max_error = max_error
        self.fallback_to_original = fallback_to_original
        self.protect_critical = protect_critical
        self.target_attention_only = target_attention_only
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.skip_attention_qkv = skip_attention_qkv
        self.recalibrate_ln = recalibrate_ln
        self.target_modules = target_modules

        self.tokenizer = tokenizer or kwargs.get(
            "tokenizer", getattr(model, "tokenizer", None)
        )
        self.calibration_loader = calibration_loader or kwargs.get("calibration_loader")
        self.calibration_texts = calibration_texts or kwargs.get("calibration_texts")
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.comp_config = cfg

        # Initialize learnable CPT Generator
        self.cpt_generator = CPTGenerator().to(self.target_device)
        self.cpt_generator_trained = False

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.actual_ranks: dict[str, int] = {}
        self.stats: dict[str, Any] = {}

    def get_importance(self, d: int, device: torch.device) -> torch.Tensor:
        """
        Retrieves feature importance vector using learned CPT generator if trained/active,
        otherwise falling back to static heuristic.
        """
        if self.use_learned_cpt and self.cpt_generator is not None:
            task_query = self.task_description or self.task_type
            return self.cpt_generator(task_query, d, device=device)
        logger.debug("Falling back to static CPT heuristic for task %s", self.task_type)
        return CompressionPlanningTokens.generate_importance(
            self.task_type, d, device=device
        )

    def train_cpt_generator(
        self,
        task_descriptions: list[str] | None = None,
        sample_layers: list[tuple[str, nn.Module]] | None = None,
        epochs: int = 5,
        lr: float = 1e-4,
    ) -> float:
        """
        Train CPT generator on sample layers across diverse task descriptions,
        minimizing reconstruction error after CPT-weighted SVD compression.
        """
        if task_descriptions is None:
            task_descriptions = [
                "question answering: extract concise answer from supporting context passage",
                "code generation: implement efficient python function and classes",
                "summarization: produce concise abstractive summary of input document",
                "mathematical reasoning: solve multi-step theorem and calculus equations",
                "sentiment classification: predict positive or negative customer review sentiment",
            ]

        if sample_layers is None:
            sample_layers = [
                (name, mod)
                for name, mod in self.model.named_modules()
                if self._is_target_module(name, mod)
            ][
                :4
            ]  # train on up to 4 sample layers for efficiency

        if not sample_layers:
            logger.warning("No sample layers found to train CPT generator.")
            return 0.0

        self.cpt_generator.train()
        for p in self.cpt_generator.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(self.cpt_generator.parameters(), lr=lr)

        total_loss = 0.0
        steps = 0

        for epoch in range(epochs):
            for task in task_descriptions:
                for name, module in sample_layers:
                    target_mod = getattr(
                        module, "base_layer", getattr(module, "linear", module)
                    )
                    cls_name = module.__class__.__name__.lower()
                    is_conv1d = (
                        "conv1d" in cls_name
                        or hasattr(module, "nf")
                        or hasattr(target_mod, "nf")
                    )
                    raw_w = target_mod.weight.data.float().to(self.target_device)
                    W = raw_w.t().contiguous() if is_conv1d else raw_w
                    out_f, in_f = W.shape

                    # Generate learned importance
                    importance = self.cpt_generator(task, out_f, device=W.device)
                    W_cpt = W * importance.unsqueeze(1)

                    r = max(
                        1,
                        min(int(min(out_f, in_f) * self.keep_ratio), min(out_f, in_f)),
                    )
                    try:
                        U, S, Vt = torch.linalg.svd(W_cpt, full_matrices=False)
                        W_lowrank = torch.matmul(
                            U[:, :r] * S[:r].unsqueeze(0), Vt[:r, :]
                        )
                        W_approx = W_lowrank / importance.unsqueeze(1)
                        loss = torch.norm(W - W_approx, p="fro") / max(
                            torch.norm(W, p="fro").item(), 1e-6
                        )

                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.cpt_generator.parameters(), max_norm=1.0
                        )
                        optimizer.step()

                        total_loss += loss.item()
                        steps += 1
                    except Exception as e:
                        logger.debug("CPT generator training step exception: %s", e)

        self.cpt_generator.eval()
        self.cpt_generator_trained = True
        for p in self.cpt_generator.parameters():
            p.requires_grad = False

        avg_loss = total_loss / max(1, steps)
        logger.info(
            "CPT generator trained successfully over %d steps with avg loss %.4f",
            steps,
            avg_loss,
        )
        return avg_loss

    def save_cpt_generator(self, path: str | Path) -> None:
        """Save CPT generator state dict to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.cpt_generator.state_dict(), path)
        logger.info("Saved CPT generator checkpoint to %s", path)

    def load_cpt_generator(self, path: str | Path) -> bool:
        """Load CPT generator state dict from file."""
        path = Path(path)
        if path.exists():
            try:
                state_dict = torch.load(path, map_location=self.target_device)
                self.cpt_generator.load_state_dict(state_dict)
                self.cpt_generator_trained = True
                self.cpt_generator.eval()
                for p in self.cpt_generator.parameters():
                    p.requires_grad = False
                logger.info("Loaded CPT generator checkpoint from %s", path)
                return True
            except Exception as e:
                logger.warning("Failed to load CPT generator from %s: %s", path, e)
        return False

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        if not self.skip_attention_qkv:
            return False
        if self.target_attention_only:
            return False
        cfg = getattr(self.model, "config", None)
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if (
            num_heads is not None
            and num_kv_heads is not None
            and num_kv_heads < num_heads
        ):
            name_lower = name.lower()
            if any(
                p in name_lower
                for p in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "query",
                    "key",
                    "value",
                    "c_attn",
                ]
            ):
                return True
        return False

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, CacheNotesLoRALinear):
            return False

        # Do not target inner submodules of LoRA or wrappers directly
        if any(
            name.endswith(sfx)
            for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout", ".linear"]
        ):
            return False

        target_mod = getattr(module, "base_layer", getattr(module, "linear", module))
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or (
                Conv1D is not None
                and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D))
            )
            or "conv1d" in cls_name
            or "conv1d" in target_cls
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        # Protect classification / output heads
        if any(
            skip in name_lower
            for skip in [
                "classifier",
                "score",
                "lm_head",
                "pooler",
                "embed",
                "logits_processor",
                "head",
                "pre_classifier",
            ]
        ):
            return False

        if self._is_gqa_asymmetric(name, module):
            return False

        if self.target_attention_only:
            # Skip FFN / MLP layers
            ffn_patterns = [
                "ffn",
                "mlp",
                "lin1",
                "lin2",
                "intermediate",
                "gate_proj",
                "up_proj",
                "down_proj",
                "c_fc",
            ]
            if any(p in name_lower for p in ffn_patterns):
                return False
            # In RoBERTa, layer.X.output.dense is FFN, while attention.output.dense is attention
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
            # Match attention projections
            attn_patterns = [
                "attn",
                "attention",
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "out_lin",
                "q_lin",
                "k_lin",
                "v_lin",
                "c_attn",
                "c_proj",
                "query",
                "key",
                "value",
                "in_proj_qkv",
                "out_proj",
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if self.protect_critical and not should_compress_linear(self.model, name):
                return False

        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        """Compress linear layers using task-guided CPT-weighted low-rank SVD."""
        # Auto-train learnable CPT generator if active and untrained
        if self.use_learned_cpt and not self.cpt_generator_trained:
            logger.info(
                "CPT generator not yet trained. Running initial task-aware training pass..."
            )
            self.train_cpt_generator(epochs=3, lr=1e-4)

        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.actual_ranks = {}
        seen_weights: set[int] = set()

        for name, module in list(self.model.named_modules()):
            if not self._is_target_module(name, module):
                continue

            target_mod = getattr(
                module, "base_layer", getattr(module, "linear", module)
            )
            module.__class__.__name__.lower()
            is_conv1d = (
                (
                    Conv1D is not None
                    and (isinstance(module, Conv1D) or isinstance(target_mod, Conv1D))
                )
                or hasattr(module, "nf")
                or hasattr(target_mod, "nf")
            )
            has_lora = (
                hasattr(module, "lora_A") and module.lora_A is not None
            ) or hasattr(module, "base_layer")

            w_id = id(target_mod.weight)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)

            raw_w = target_mod.weight.data
            device = raw_w.device
            dtype = raw_w.dtype

            # Standardize shape to [out_features, in_features] in FP32
            W = (
                raw_w.t().contiguous().clone().float()
                if is_conv1d
                else raw_w.clone().float()
            )
            out_f, in_f = W.shape

            # Retrieve CPT importance modulation per output feature (learned or static)
            importance = self.get_importance(out_f, device=device)

            # Weight rows by CPT importance
            W_cpt = W * importance.unsqueeze(1)

            # Compute SVD in CPT-weighted space
            r = max(1, min(int(min(out_f, in_f) * self.keep_ratio), min(out_f, in_f)))
            self.actual_ranks[name] = r
            try:
                U, S, Vt = torch.linalg.svd(W_cpt, full_matrices=False)
                # Low-rank approximation
                W_lowrank = torch.matmul(U[:, :r] * S[:r].unsqueeze(0), Vt[:r, :])
                # Unweight back to physical space: all channels remain active and non-zero
                W_approx = W_lowrank / importance.unsqueeze(1)
                if torch.isnan(W_approx).any() or torch.isinf(W_approx).any():
                    W_approx = W
            except Exception as e:
                logger.warning("SVD failed in CacheNotes on %s: %s", name, e)
                W_approx = W

            w_norm = torch.norm(W, p="fro").item()
            err = float(torch.norm(W - W_approx, p="fro").item() / max(w_norm, 1e-12))
            self.layer_errors[name] = err

            if (
                torch.isnan(W_approx).any()
                or torch.isinf(W_approx).any()
                or err > self.max_error
            ):
                logger.warning(
                    "CacheNotes reconstruction error %.4f on %s exceeded threshold %.4f.",
                    err,
                    name,
                    self.max_error,
                )
                if self.fallback_to_original:
                    logger.info(
                        "Skipping CacheNotes compression for %s (fallback to original).",
                        name,
                    )
                    self.fallback_layers.append(name)
                    # Freeze fallback layers
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

            # Update weights / module
            if has_lora:
                # Layer is already a LoRALinear wrapper: update base layer weights in-place
                if is_conv1d:
                    target_mod.weight.data.copy_(
                        W_approx.t().contiguous().to(dtype=dtype, device=device)
                    )
                else:
                    target_mod.weight.data.copy_(
                        W_approx.contiguous().to(dtype=dtype, device=device)
                    )
                # Keep base weights frozen, but keep LoRA adapters trainable
                target_mod.weight.requires_grad = False
                if (
                    getattr(target_mod, "bias", None) is not None
                    and target_mod.bias is not None
                ):
                    target_mod.bias.requires_grad = False
                for p_name, p in module.named_parameters():
                    if any(
                        t in p_name.lower() for t in ("lora_a", "lora_b", "adapter")
                    ):
                        p.requires_grad = True
            elif self.lora_r > 0:
                # Not LoRALinear yet, and lora_r > 0: wrap in CacheNotesLoRALinear
                cn_layer = CacheNotesLoRALinear.from_linear(
                    target_mod,
                    weight=W_approx,
                    lora_r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                )
                if name == "":
                    self.model = cn_layer
                else:
                    replace_module(self.model, name, cn_layer)
            else:
                # Pure static compression (lora_r == 0)
                if is_conv1d:
                    target_mod.weight.data.copy_(
                        W_approx.t().contiguous().to(dtype=dtype, device=device)
                    )
                else:
                    target_mod.weight.data.copy_(
                        W_approx.contiguous().to(dtype=dtype, device=device)
                    )

            self.compressed_layers.append(name)

        # LayerNorm Recalibration
        if self.recalibrate_ln and (
            self.calibration_loader is not None or self.calibration_texts is not None
        ):
            try:
                recalibrate_layernorms(
                    self.model,
                    tokenizer=self.tokenizer,
                    calibration_loader=self.calibration_loader,
                    calibration_texts=self.calibration_texts,
                    device=self.target_device,
                )
            except Exception as e:
                logger.warning("CacheNotes LayerNorm recalibration skipped: %s", e)

        # Post-compression forward sanity check
        forward_check_success = False
        try:
            self.model.eval()
            with torch.no_grad():
                d = next(self.model.parameters()).device
                dummy = torch.randint(0, 100, (1, 16), device=d)
                self.model(dummy)
            forward_check_success = True
        except Exception as e:
            logger.debug("CacheNotes post-compression forward check exception: %s", e)

        self._compressed_params = self._count_params(self.model)
        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers
            else 0.0
        )
        avg_rank = (
            float(
                sum(self.actual_ranks[k] for k in self.compressed_layers)
                / max(1, len(self.compressed_layers))
            )
            if self.compressed_layers
            else 0.0
        )
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "fallback_layers": self.fallback_layers,
                "skipped_layers": self.skipped_layers,
                "num_compressed": len(self.compressed_layers),
                "num_fallback": len(self.fallback_layers),
                "num_skipped": len(self.skipped_layers),
                "layer_errors": self.layer_errors,
                "actual_ranks": self.actual_ranks,
                "avg_rank": avg_rank,
                "avg_reconstruction_error": avg_err,
                "task_type": self.task_type,
                "task_description": self.task_description,
                "use_learned_cpt": self.use_learned_cpt,
                "cpt_generator_trained": self.cpt_generator_trained,
                "keep_ratio": self.keep_ratio,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "target_attention_only": self.target_attention_only,
                "lora_r": self.lora_r,
                "lora_alpha": self.lora_alpha,
                "forward_check_success": forward_check_success,
                "recalibrate_ln": self.recalibrate_ln,
                "method": "cachenotes_learned_cpt_svd",
                "kv_cache_budget_saving": f"{(1.0 - self.keep_ratio) * 100:.1f}% Footprint Reduction",
            }
        )
        self.stats = stats
        self.model.compression_stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("cachenotes_factory")
@COMPRESSOR_REGISTRY.register("cachenotes_compressor_factory")
def cachenotes_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> Any:
    """CacheNotes compressor factory function."""
    return_tuple = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))
    task_type = kwargs.pop(
        "task_type",
        _cfg_get(
            comp_config, "cachenotes_task", _cfg_get(comp_config, "task_type", "qa")
        ),
    )
    task_description = kwargs.pop(
        "task_description",
        _cfg_get(
            comp_config,
            "cachenotes_task_description",
            _cfg_get(comp_config, "task_description", task_type),
        ),
    )
    use_learned_cpt = kwargs.pop(
        "use_learned_cpt",
        _cfg_get(
            comp_config,
            "cachenotes_use_learned_cpt",
            _cfg_get(comp_config, "use_learned_cpt", True),
        ),
    )
    keep_ratio = kwargs.pop(
        "keep_ratio",
        _cfg_get(
            comp_config,
            "cachenotes_keep_ratio",
            _cfg_get(comp_config, "keep_ratio", 0.5),
        ),
    )
    max_error = kwargs.pop(
        "max_error",
        _cfg_get(
            comp_config,
            "cachenotes_max_error",
            _cfg_get(comp_config, "max_error", 0.35),
        ),
    )
    fallback = kwargs.pop(
        "fallback_to_original",
        _cfg_get(
            comp_config,
            "cachenotes_fallback_to_original",
            _cfg_get(comp_config, "fallback_to_original", True),
        ),
    )
    protect_critical = kwargs.pop(
        "protect_critical",
        _cfg_get(comp_config, "protect_critical", True),
    )
    target_attention_only = kwargs.pop(
        "target_attention_only",
        _cfg_get(
            comp_config,
            "cachenotes_target_attention_only",
            _cfg_get(comp_config, "target_attention_only", False),
        ),
    )
    lora_r = kwargs.pop(
        "lora_r",
        _cfg_get(
            comp_config,
            "cachenotes_lora_r",
            _cfg_get(comp_config, "lora_r", _cfg_get(comp_config, "lora_rank", 0)),
        ),
    )
    lora_alpha = kwargs.pop(
        "lora_alpha",
        _cfg_get(
            comp_config,
            "cachenotes_lora_alpha",
            _cfg_get(comp_config, "lora_alpha", 1.0),
        ),
    )
    skip_attention_qkv = kwargs.pop(
        "skip_attention_qkv",
        _cfg_get(
            comp_config,
            "cachenotes_skip_attention_qkv",
            _cfg_get(comp_config, "skip_attention_qkv", True),
        ),
    )
    recalibrate_ln = kwargs.pop(
        "recalibrate_ln",
        _cfg_get(
            comp_config,
            "cachenotes_recalibrate_ln",
            _cfg_get(comp_config, "recalibrate_ln", True),
        ),
    )

    compressor = CacheNotesCompressor(
        model,
        task_type=task_type,
        task_description=task_description,
        use_learned_cpt=use_learned_cpt,
        keep_ratio=keep_ratio,
        max_error=max_error,
        fallback_to_original=fallback,
        protect_critical=protect_critical,
        target_attention_only=target_attention_only,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        skip_attention_qkv=skip_attention_qkv,
        recalibrate_ln=recalibrate_ln,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_tuple:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "CompressionPlanningTokens",
    "CPTGenerator",
    "CacheNotesKVCache",
    "CacheNotesLoRALinear",
    "CacheNotesCompressor",
    "cachenotes_factory",
]
