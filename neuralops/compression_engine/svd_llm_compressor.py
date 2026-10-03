"""
SVD-LLM Compressor: Truncation-Aware Singular Value Decomposition for LLM Compression (ICLR 2025).
Paper Reference: Wang et al., "SVD-LLM: Truncation-Aware Singular Value Decomposition for Large Language Model Compression", ICLR 2025 (arXiv:2403.07378).

Features implemented faithfully according to the paper:
1. Data-Driven Cholesky Activation Whitening:
   - Accumulate input activation covariance C = (1/N) X^T X using calibration data
   - Cholesky factorization S = cholesky(C + eps * I)
   - Whiten weight: W_white = W @ S
   - Truncated SVD: W_white ≈ U_r @ diag(Sigma_r) @ Vt_r
   - Unwhiten right factor: Vt_unw = Vt_r @ S^{-1}
   - Explicit factor matrices: W_u = U_r * sqrt(Sigma_r), W_v = sqrt(Sigma_r) * Vt_unw
   - Forward mapping: y = (x @ W_v.T) @ W_u.T + bias
2. Sequential LoRA Fine-Tuning (Section 3.2, Eq. 7):
   - Stage 1: Update W_u via LoRA (freeze W_v)
   - Stage 2: Update W_v via LoRA (freeze W_u)
3. Fixed Parameter Rank Ratio:
   - Given parameter ratio R ∈ (0, 1]: r = (R * m * n) / (m + n)
4. Compatibility across DistilBERT, RoBERTa/BERT, GPT-2 (Conv1D), and LLaMA/Qwen.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralops.core.base import BaseCompressor
from neuralops.core.model_utils import replace_module
from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)

# Standard multi-domain calibration corpus (shared with AASVD for direct parity)
DEFAULT_CALIBRATION_CORPUS: list[str] = [
    (
        "Transformers rely heavily on multi-head self-attention mechanisms to capture complex syntactic "
        "and semantic dependencies across arbitrary sequence lengths. However, the standard quadratic computational "
        "complexity of softmax attention poses significant computational and memory bandwidth challenges during "
        "both pre-training and long-context autoregressive decoding."
    ),
    (
        "Singular Value Decomposition provides an optimal low-rank subspace approximation under the Frobenius norm. "
        "By preconditioning the weight matrices with empirical input activation covariances, activation-aware SVD "
        "preserves salient outlier directions that prevent catastrophic representation drift in deep neural networks."
    ),
    (
        "def compute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:\n"
        "    scores = torch.matmul(q, k.transpose(-1, -2).contiguous()) * scale\n"
        "    probs = torch.softmax(scores, dim=-1)\n"
        "    return torch.matmul(probs, v)\n"
    ),
    (
        "In distributed multi-GPU environments, tensor parallelism and fully sharded data parallelism require "
        "efficient parameter sharding. Low-rank factorized layers reduce inter-node communication volume by communicating "
        "narrow bottleneck activations rather than high-dimensional hidden states across all-reduce collectives."
    ),
    (
        "Mathematical problem solving and chain-of-thought reasoning require high-fidelity preservation of intermediate "
        "residual stream representations. Let f(x) = Wx + b. If W is factorized into U and V with rank r << min(d_in, d_out), "
        "the total number of parameters scales as r*(d_in + d_out) compared to d_in*d_out."
    ),
]


class SVDLLMLinear(nn.Module):
    """
    SVD-LLM Factorized Linear Layer executing as:
    y = (x @ W_v.T + lora_v(x)) @ (W_u.T) + lora_u(x @ W_v.T) + bias

    Specifically, with sequential LoRA support:
    - W_u ∈ R^(d_out × r), W_v ∈ R^(r × d_in)
    - LoRA for W_u: A_u ∈ R^(r_L × r), B_u ∈ R^(d_out × r_L)
    - LoRA for W_v: A_v ∈ R^(r_L × d_in), B_v ∈ R^(r × r_L)
    """

    is_custom_attention: bool = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(1, lora_rank)

        # Explicit factor buffers: W_u and W_v
        self.W_u = nn.Parameter(
            torch.empty(out_features, self.rank, device=device, dtype=dtype)
        )
        self.W_v = nn.Parameter(
            torch.empty(self.rank, in_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        # LoRA adapters for W_u and W_v
        if self.lora_rank > 0:
            # LoRA for W_u: input dim = r, output dim = out_features
            self.lora_A_u = nn.Parameter(
                torch.empty(lora_rank, self.rank, device=device, dtype=dtype)
            )
            self.lora_B_u = nn.Parameter(
                torch.empty(out_features, lora_rank, device=device, dtype=dtype)
            )

            # LoRA for W_v: input dim = in_features, output dim = r
            self.lora_A_v = nn.Parameter(
                torch.empty(lora_rank, in_features, device=device, dtype=dtype)
            )
            self.lora_B_v = nn.Parameter(
                torch.empty(self.rank, lora_rank, device=device, dtype=dtype)
            )

            self.reset_lora_parameters()
        else:
            self.register_parameter("lora_A_u", None)
            self.register_parameter("lora_B_u", None)
            self.register_parameter("lora_A_v", None)
            self.register_parameter("lora_B_v", None)

        self.reset_base_parameters()

    def reset_base_parameters(self) -> None:
        with torch.no_grad():
            nn.init.orthogonal_(self.W_u)
            nn.init.orthogonal_(self.W_v)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def reset_lora_parameters(self) -> None:
        with torch.no_grad():
            if self.lora_rank > 0:
                nn.init.kaiming_uniform_(self.lora_A_u, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_u)
                nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B_v)

    def set_factors(
        self, W_u: torch.Tensor, W_v: torch.Tensor, bias: torch.Tensor | None = None
    ) -> None:
        with torch.no_grad():
            r = min(self.rank, W_u.size(1), W_v.size(0))
            self.W_u.data.zero_()
            self.W_v.data.zero_()
            self.W_u.data[:, :r].copy_(
                W_u[:, :r].to(device=self.W_u.device, dtype=self.W_u.dtype)
            )
            self.W_v.data[:r, :].copy_(
                W_v[:r, :].to(device=self.W_v.device, dtype=self.W_v.dtype)
            )
            if bias is not None and self.bias is not None:
                self.bias.data.copy_(
                    bias.to(device=self.bias.device, dtype=self.bias.dtype)
                )

    def set_active_lora_stage(self, stage: str | None) -> None:
        """
        stage == 'stage1': train A_u, B_u only (W_v frozen)
        stage == 'stage2': train A_v, B_v only (W_u frozen)
        stage == 'both': train all LoRA adapters
        stage is None: freeze all LoRA adapters
        """
        self.W_u.requires_grad = False
        self.W_v.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        if self.lora_rank == 0:
            return

        if stage == "stage1":
            self.lora_A_u.requires_grad = True
            self.lora_B_u.requires_grad = True
            self.lora_A_v.requires_grad = False
            self.lora_B_v.requires_grad = False
        elif stage == "stage2":
            self.lora_A_u.requires_grad = False
            self.lora_B_u.requires_grad = False
            self.lora_A_v.requires_grad = True
            self.lora_B_v.requires_grad = True
        elif stage == "both":
            self.lora_A_u.requires_grad = True
            self.lora_B_u.requires_grad = True
            self.lora_A_v.requires_grad = True
            self.lora_B_v.requires_grad = True
        else:
            self.lora_A_u.requires_grad = False
            self.lora_B_u.requires_grad = False
            self.lora_A_v.requires_grad = False
            self.lora_B_v.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Compute latent representation through W_v
        # latent = x @ W_v.T + (optional LoRA_v)
        latent = F.linear(x, self.W_v)
        if (
            self.lora_rank > 0
            and self.lora_A_v is not None
            and self.lora_B_v is not None
        ):
            lora_v = (
                F.linear(
                    F.linear(x.to(self.lora_A_v.dtype), self.lora_A_v), self.lora_B_v
                )
                * self.scaling
            )
            latent = latent + lora_v.to(latent.dtype)

        # 2. Compute output through W_u
        # out = latent @ W_u.T + (optional LoRA_u)
        out = F.linear(latent, self.W_u)
        if (
            self.lora_rank > 0
            and self.lora_A_u is not None
            and self.lora_B_u is not None
        ):
            lora_u = (
                F.linear(
                    F.linear(latent.to(self.lora_A_u.dtype), self.lora_A_u),
                    self.lora_B_u,
                )
                * self.scaling
            )
            out = out + lora_u.to(out.dtype)

        if self.bias is not None:
            out = out + self.bias
        return out


@COMPRESSOR_REGISTRY.register("svd_llm")
@COMPRESSOR_REGISTRY.register("svd_llm_compressor")
class SVDLLMCompressor(BaseCompressor):
    """
    SVD-LLM (ICLR 2025) Compressor with Cholesky activation whitening,
    sequential LoRA recovery fine-tuning, and fixed rank parameter ratios.
    """

    def __init__(
        self,
        model: nn.Module,
        rank_ratio: float = 0.20,
        rank: int | None = None,
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        max_error: float = 0.50,
        fallback_to_original: bool = False,
        calibration_texts: list[str] | None = None,
        tokenizer: Any | None = None,
        device: str | torch.device | None = None,
        target_attention_only: bool = True,
        recovery_steps: int = 50,
        learning_rate: float = 2e-4,
        weight_decay: float = 0.01,
        batch_size: int = 8,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, name="svd_llm", **kwargs)
        self.rank_ratio = rank_ratio
        self.explicit_rank = rank
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.max_error = max_error
        self.fallback_to_original = fallback_to_original
        self.target_attention_only = target_attention_only
        self.recovery_steps = recovery_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size

        if device is not None:
            self.device = torch.device(device)
        else:
            p = next(model.parameters(), None)
            self.device = p.device if p is not None else torch.device("cpu")

        self.tokenizer = tokenizer
        if calibration_texts is not None:
            self.calibration_texts = calibration_texts
        else:
            self.calibration_texts = self._load_calibration_data(n_samples=256)

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.layer_ranks: dict[str, int] = {}

    def _load_calibration_data(self, n_samples: int = 256) -> list[str]:
        """
        Load calibration samples from WikiText-2 (or Salesforce/wikitext fallback),
        following SVD-LLM (ICLR 2025) paper protocol. Falls back to default corpus if offline.
        """
        try:
            from datasets import load_dataset

            try:
                ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            except Exception:
                ds = load_dataset(
                    "Salesforce/wikitext", "wikitext-2-raw-v1", split="train"
                )
            texts = [t.strip() for t in ds["text"] if t and len(t.strip()) > 50]
            if len(texts) >= n_samples:
                logger.info(
                    "Loaded %d calibration samples from WikiText-2 for SVD-LLM.",
                    n_samples,
                )
                return texts[:n_samples]
            elif texts:
                return texts
        except Exception as e:
            logger.warning(
                "Could not download WikiText-2 calibration dataset (%s); using default corpus.",
                e,
            )
        return DEFAULT_CALIBRATION_CORPUS

    def compute_rank_from_ratio(self, m: int, n: int, ratio: float) -> int:
        """
        Compute low rank r from parameter retention ratio R:
        Original params = m * n
        Target params = R * m * n
        Factorized params = r * (m + n)
        r = (R * m * n) / (m + n)
        """
        if self.explicit_rank is not None and self.explicit_rank > 0:
            return min(self.explicit_rank, min(m, n))
        r_calc = (ratio * m * n) / (m + n)
        r = int(round(r_calc))
        return max(1, min(r, min(m, n)))

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, SVDLLMLinear):
            return False
        if any(
            name.endswith(sfx)
            for sfx in [".base_layer", ".lora_A", ".lora_B", ".lora_dropout"]
        ):
            return False

        target_mod = getattr(module, "base_layer", module)
        cls_name = module.__class__.__name__.lower()
        target_cls = target_mod.__class__.__name__.lower()

        is_linear_like = (
            isinstance(module, nn.Linear)
            or isinstance(target_mod, nn.Linear)
            or "conv1d" in cls_name
            or "conv1d" in target_cls
            or hasattr(module, "nf")
            or hasattr(target_mod, "nf")
        )
        if not is_linear_like:
            return False

        name_lower = name.lower()
        if any(
            skip in name_lower
            for skip in [
                "lm_head",
                "classifier",
                "head",
                "output_layer",
                "score",
                "pooler",
            ]
        ):
            return False

        if self.target_attention_only:
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
            if "output.dense" in name_lower and "attention" not in name_lower:
                return False
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
            ]
            if not any(p in name_lower for p in attn_patterns):
                return False
        else:
            if not should_compress_linear(self.model, name):
                return False

        return True

    def _collect_covariances(self) -> dict[str, torch.Tensor]:
        """
        Collect empirical input activation covariances C = (1/N) X^T X via forward hooks.
        Uses streaming accumulation to avoid storing large activation buffers in memory.
        """
        covariances: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: tuple[torch.Tensor, ...], outp: Any):
                if not inp or inp[0] is None:
                    return
                x = inp[0].detach()
                dev_type = x.device.type
                autocast_ctx = (
                    torch.autocast(device_type=dev_type, enabled=False)
                    if hasattr(torch, "autocast") and dev_type in ("cuda", "cpu")
                    else contextlib.nullcontext()
                )
                with autocast_ctx:
                    target_mod = getattr(module, "base_layer", module)
                    cls_name = module.__class__.__name__.lower()
                    target_cls = target_mod.__class__.__name__.lower()
                    is_conv1d = (
                        "conv1d" in cls_name
                        or "conv1d" in target_cls
                        or hasattr(module, "nf")
                        or hasattr(target_mod, "nf")
                    )
                    weight_tensor = getattr(
                        target_mod, "weight", getattr(module, "weight", None)
                    )
                    if weight_tensor is None:
                        return
                    in_f = (
                        weight_tensor.shape[0] if is_conv1d else weight_tensor.shape[1]
                    )
                    x_fp32 = x.float()
                    x_flat = x_fp32.reshape(-1, in_f)
                    mask = x_flat.abs().sum(dim=-1) > 1e-5
                    if mask.sum() == 0:
                        return
                    x_active = x_flat[mask]
                    batch_cov = torch.matmul(x_active.T, x_active).cpu()
                    n_samples = x_active.size(0)

                if name not in covariances:
                    covariances[name] = batch_cov
                    counts[name] = n_samples
                else:
                    covariances[name] += batch_cov
                    counts[name] += n_samples

            return hook

        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                hooks.append(module.register_forward_hook(make_hook(name)))

        if self.tokenizer is not None and self.calibration_texts:
            try:
                inputs = self.tokenizer(
                    self.calibration_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    self.model(**inputs)
            except Exception as e:
                logger.warning("SVD-LLM calibration forward pass exception: %s", e)

        for h in hooks:
            h.remove()

        final_cov: dict[str, torch.Tensor] = {}
        for name, cov in covariances.items():
            n = max(1, counts.get(name, 1))
            final_cov[name] = cov / float(n)

        return final_cov

    def _robust_cholesky(self, C: torch.Tensor, n: int) -> torch.Tensor:
        """
        Compute Cholesky factor S where S = cholesky(C + eps * I).
        Paper specification: eps = 1e-6 * mean(diag(C))
        Uses adaptive regularisation if C is singular or ill-conditioned.
        """
        mean_diag = torch.mean(torch.diag(C)).item()
        for exp in [-6, -5, -4, -3, -2, -1, 0]:
            eps = max(10**exp * mean_diag, 10**exp)
            try:
                S = torch.linalg.cholesky(
                    C + eps * torch.eye(n, device=C.device, dtype=C.dtype)
                )
                return S
            except Exception:
                continue

        # Eigen-fallback for extreme singularity
        eigvals, eigvecs = torch.linalg.eigh(C)
        eigvals = torch.clamp(eigvals, min=1e-6)
        return eigvecs @ torch.diag(torch.sqrt(eigvals))

    def _cholesky_whiten_and_decompose(
        self,
        W: torch.Tensor,
        C: torch.Tensor | None,
        target_rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Full SVD-LLM core algorithm (Section 3.1):
        1. S = cholesky(C + eps * I)
        2. W_white = W @ S
        3. U, Sigma, Vt = svd(W_white)
        4. Truncate to rank r
        5. Vt_unw = Vt_r @ inv(S)
        6. W_u = U_r * sqrt(Sigma_r), W_v = sqrt(Sigma_r) * Vt_unw
        7. Relative Frobenius error: ||W - W_u @ W_v||_F / ||W||_F
        """
        m, n = W.shape
        W_fp32 = W.float()

        if C is not None and C.shape == (n, n):
            C_fp32 = C.float().to(device=W.device)
            C_sym = 0.5 * (C_fp32 + C_fp32.T)
            S = self._robust_cholesky(C_sym, n)
            try:
                S_inv = torch.linalg.pinv(S)
            except Exception:
                S_inv = torch.inverse(
                    S + 1e-5 * torch.eye(n, device=S.device, dtype=S.dtype)
                )
            W_white = W_fp32 @ S
        else:
            # Identity whitening (standard SVD fallback if covariance unavailable)
            S = torch.eye(n, device=W.device, dtype=W_fp32.dtype)
            S_inv = torch.eye(n, device=W.device, dtype=W_fp32.dtype)
            W_white = W_fp32

        U, Sigma, Vt = torch.linalg.svd(W_white, full_matrices=False)
        r = min(target_rank, min(m, n), U.size(1), Vt.size(0))

        U_r = U[:, :r]
        Sigma_r = Sigma[:r]
        Vt_r = Vt[:r, :]

        Vt_unw = Vt_r @ S_inv
        sqrt_sigma = torch.sqrt(torch.clamp(Sigma_r, min=1e-12))
        W_u = U_r * sqrt_sigma.unsqueeze(0)
        W_v = sqrt_sigma.unsqueeze(1) * Vt_unw

        W_rec = W_u @ W_v
        orig_norm = torch.norm(W_fp32, p="fro").clamp(min=1e-12)
        error = (torch.norm(W_fp32 - W_rec, p="fro") / orig_norm).item()

        return W_u, W_v, error

    def _compute_recovery_loss(
        self, batch_ids: torch.Tensor, batch_mask: torch.Tensor
    ) -> torch.Tensor | None:
        """
        Compute recovery training loss across both CausalLM and Sequence Classification architectures.
        - CausalLM (GPT-2, LLaMA): cross-entropy autoregressive loss with labels=batch_ids.
        - Sequence Classification (DistilBERT, RoBERTa): unsupervised entropy minimization over predicted logits.
        """
        try:
            # 1. CausalLM path
            if (
                hasattr(self.model, "lm_head")
                or "causallm" in self.model.__class__.__name__.lower()
            ):
                outputs = self.model(
                    input_ids=batch_ids, attention_mask=batch_mask, labels=batch_ids
                )
                loss = outputs.loss if hasattr(outputs, "loss") else None
                if loss is None and isinstance(outputs, tuple):
                    loss = outputs[0]
                return loss

            # 2. Sequence Classification / generic encoder path: entropy minimization
            outputs = self.model(input_ids=batch_ids, attention_mask=batch_mask)
            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, tuple):
                logits = outputs[0]
            if (
                logits is not None
                and isinstance(logits, torch.Tensor)
                and logits.ndim >= 2
            ):
                probs = torch.softmax(logits.float(), dim=-1)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
                return entropy
            return None
        except Exception as e:
            logger.debug("Error computing recovery loss: %s", e)
            return None

    def _sequential_lora_recovery(self) -> None:
        """
        Execute 2-Stage Sequential LoRA fine-tuning (Section 3.2, Eq. 7):
        Stage 1: Update W_u with LoRA (W_v frozen)
        Stage 2: Update W_v with LoRA (W_u frozen)
        """
        if self.recovery_steps <= 0 or self.lora_rank <= 0:
            return
        if self.tokenizer is None or not self.calibration_texts:
            return

        logger.info(
            "[SVD-LLM] Starting 2-Stage Sequential LoRA Recovery (%d steps/stage)...",
            self.recovery_steps,
        )
        device = self.device

        # Step 0: Freeze ALL model parameters (embeddings, layernorms, heads, and base weights)
        for p in self.model.parameters():
            p.requires_grad = False

        # Prepare calibration DataLoader
        encodings = self.tokenizer(
            self.calibration_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        dataset = torch.utils.data.TensorDataset(
            encodings["input_ids"], encodings["attention_mask"]
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True
        )

        for stage, stage_name in [
            ("stage1", "Stage 1 (Update W_u)"),
            ("stage2", "Stage 2 (Update W_v)"),
        ]:
            logger.info(f"[SVD-LLM Sequential LoRA] {stage_name}...")
            # Set active LoRA adapters
            for m in self.model.modules():
                if isinstance(m, SVDLLMLinear):
                    m.set_active_lora_stage(stage)

            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            if not trainable_params:
                continue

            optimizer = torch.optim.AdamW(
                trainable_params, lr=self.learning_rate, weight_decay=self.weight_decay
            )
            step = 0
            self.model.train()

            while step < self.recovery_steps:
                for batch_ids, batch_mask in loader:
                    if step >= self.recovery_steps:
                        break
                    batch_ids = batch_ids.to(device)
                    batch_mask = batch_mask.to(device)

                    optimizer.zero_grad()
                    loss = self._compute_recovery_loss(batch_ids, batch_mask)

                    if loss is not None and not torch.isnan(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                        optimizer.step()
                    step += 1

        # Freeze all adapters after recovery
        for p in self.model.parameters():
            p.requires_grad = False
        for m in self.model.modules():
            if isinstance(m, SVDLLMLinear):
                m.set_active_lora_stage(None)
        self.model.eval()
        logger.info("[SVD-LLM] Sequential LoRA Recovery Complete.")

    def compress(self) -> tuple[nn.Module, dict[str, Any]]:
        """
        Compress the target model using full SVD-LLM Cholesky whitening and structural replacement.
        """
        covariances = self._collect_covariances()
        target_layers: list[tuple[str, nn.Module]] = []
        for name, module in self.model.named_modules():
            if self._is_target_module(name, module):
                target_layers.append((name, module))

        logger.info(
            "Found %d linear layers targeted for SVD-LLM compression.",
            len(target_layers),
        )

        for name, layer in target_layers:
            target_mod = getattr(layer, "base_layer", layer)
            cls_name = layer.__class__.__name__.lower()
            target_cls = target_mod.__class__.__name__.lower()
            is_conv1d = (
                "conv1d" in cls_name
                or "conv1d" in target_cls
                or hasattr(layer, "nf")
                or hasattr(target_mod, "nf")
            )

            weight_tensor = getattr(
                target_mod, "weight", getattr(layer, "weight", None)
            )
            if weight_tensor is None:
                continue

            bias_tensor = getattr(target_mod, "bias", getattr(layer, "bias", None))
            W = weight_tensor.data.clone()
            if is_conv1d:
                W = W.t().contiguous()

            m, n = W.shape
            r = self.compute_rank_from_ratio(m, n, self.rank_ratio)

            # Check parameter reduction condition
            if r * (m + n) >= m * n:
                self.fallback_layers.append(name)
                continue

            C = covariances.get(name, None)
            try:
                W_u, W_v, error = self._cholesky_whiten_and_decompose(
                    W, C, target_rank=r
                )
            except Exception as decomp_err:
                logger.warning(
                    "SVD-LLM decomposition failed for layer %s: %s", name, decomp_err
                )
                self.fallback_layers.append(name)
                continue

            self.layer_errors[name] = error
            self.layer_ranks[name] = r

            # In faithful SVD-LLM (ICLR 2025), fallback is not used; all target layers are compressed.
            if self.fallback_to_original and error > self.max_error:
                logger.info(
                    "SVD-LLM layer %s error %.4f > max_error %.4f; skipping (fallback).",
                    name,
                    error,
                    self.max_error,
                )
                self.fallback_layers.append(name)
                continue

            svd_layer = SVDLLMLinear(
                in_features=n,
                out_features=m,
                rank=r,
                bias=bias_tensor is not None,
                dtype=weight_tensor.dtype,
                device=weight_tensor.device,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
            )
            svd_layer.set_factors(W_u, W_v, bias_tensor)

            replace_module(self.model, name, svd_layer)
            self.compressed_layers.append(name)

        # Run sequential LoRA fine-tuning
        self._sequential_lora_recovery()

        stats = {
            "num_compressed_layers": len(self.compressed_layers),
            "num_fallback_layers": len(self.fallback_layers),
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "layer_errors": self.layer_errors,
            "layer_ranks": self.layer_ranks,
            "mean_error": float(
                sum(self.layer_errors.values()) / max(1, len(self.layer_errors))
            ),
        }
        self.model.compression_stats = stats
        return self.model, stats
