"""
Swift-SVD Activation-Aware Compressor — Production-Hardened (v2).

Registered as 'swift_svd_act', 'swift_svd', 'swift_svd_compressor'.

Algorithm:
  1. Collect empirical cross-covariance C = E[Y^T X] per layer across
     calibration data (online accumulation, bounded memory).
  2. Extract principal activation subspaces:
       U_act = top-r left  singular vectors of C
       V_act = top-r right singular vectors of C
  3. Project W onto (U_act, V_act):
       M = U_act^T @ W @ V_act
  4. SVD the projected matrix and reconstruct:
       A = U_act @ U_m · sqrt(S_m)
       B = sqrt(S_m) · Vt_m @ V_act^T
  5. Replace the dense layer with SVDLinear (or fall back to plain SVD).

"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear
from framework.core.pruning_utils import get_calibration_tensors

logger = logging.getLogger(__name__)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register("swift_svd_act")
@COMPRESSOR_REGISTRY.register("swift_svd")
@COMPRESSOR_REGISTRY.register("swift_svd_compressor")
class SwiftSVDActivationCompressor(BaseCompressor):
    """
    Swift-SVD Activation-Aware Compressor.

    Captures empirical input-output cross-covariances across calibration data,
    derives activation-aligned subspaces, projects weights onto the principal
    activation subspace, and replaces dense nn.Linear / Conv1D layers with
    factorized SVDLinear modules.
    """
    # SW-FIX-09
    is_custom_attention: bool = False

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any | None = None,
        calibration_loader: Any | None = None,
        calibration_texts: list[str] | None = None,
        rank: int = 64,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        target_attention_only: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        skip_attention_qkv: bool = False,
        min_rank: int = 4,
        target_modules: list[str] | None = None,
        num_calibration_batches: int = 8,
        max_calibration_tokens: int = 32768,
        use_fp64_cross_cov: bool = True,     # SW-FIX-16
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="swift_svd_act", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg"))
        if cfg is not None:
            rank = _cfg_get(cfg, "swift_svd_act_rank", _cfg_get(cfg, "rank", rank))
            tokenizer = _cfg_get(cfg, "tokenizer", tokenizer)
            calibration_loader = _cfg_get(cfg, "calibration_loader", calibration_loader)
            calibration_texts = _cfg_get(cfg, "calibration_texts", calibration_texts)
            max_error = _cfg_get(cfg, "max_error", max_error)
            fallback_to_original = _cfg_get(cfg, "fallback_to_original", fallback_to_original)
            target_attention_only = _cfg_get(cfg, "target_attention_only", target_attention_only)
            lora_r = _cfg_get(cfg, "lora_r", lora_r)
            lora_alpha = _cfg_get(cfg, "lora_alpha", lora_alpha)
            skip_attention_qkv = _cfg_get(cfg, "skip_attention_qkv", skip_attention_qkv)
            min_rank = _cfg_get(cfg, "min_rank", min_rank)
            num_calibration_batches = _cfg_get(cfg, "num_calibration_batches", num_calibration_batches)
            max_calibration_tokens = _cfg_get(cfg, "max_calibration_tokens", max_calibration_tokens)
            use_fp64_cross_cov = _cfg_get(cfg, "use_fp64_cross_cov", use_fp64_cross_cov)

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.tokenizer = tokenizer
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.rank = int(rank)
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.target_attention_only = bool(target_attention_only)
        self.lora_r = max(0, int(lora_r))
        self.lora_alpha = float(lora_alpha)
        self.skip_attention_qkv = bool(skip_attention_qkv)
        self.min_rank = max(1, int(min_rank))
        self.use_fp64_cross_cov = bool(use_fp64_cross_cov)

        # SW-FIX-12: empty list → None
        if target_modules is not None and len(target_modules) == 0:
            target_modules = None
        self.target_modules = target_modules

        self.num_calibration_batches = max(1, int(num_calibration_batches))
        self.max_calibration_tokens = max(1, int(max_calibration_tokens))

        self.compressed_layers: list[str] = []
        self.fallback_layers: list[str] = []
        self.skipped_layers: list[str] = []
        self.layer_errors: dict[str, float] = {}
        self.replaced_layers: list[dict[str, Any]] = []
        self.stats: dict[str, Any] = {}
        self._compressed_params: int = 0
        self._orig_storage_bytes: int = 0
        self._comp_storage_bytes: int = 0

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------
    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        # SW-FIX-11: always active, not gated by skip_attention_qkv.
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return False
        num_heads = getattr(cfg, "num_attention_heads", None)
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_heads is None or num_kv_heads is None:
            return False
        if num_kv_heads >= num_heads:
            return False
        name_lower = name.lower()
        return any(
            p in name_lower
            for p in ("q_proj", "k_proj", "v_proj", "query", "key", "value", "c_attn")
        )

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        if any(k in name_l for k in
               ("classifier", "score", "lm_head", "pooler", "embed", "logits_processor")):
            return False

        if isinstance(module, (nn.LayerNorm, nn.RMSNorm, nn.Embedding,
                                nn.BatchNorm1d, nn.BatchNorm2d)):
            return False

        if self.target_attention_only:
            attn_keywords = ('attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj',
                             'c_attn', 'c_proj', 'query', 'key', 'value')
            if not any(k in name_l for k in attn_keywords):
                return False

        base_mod = getattr(module, "base_layer", module)
        cls_name = base_mod.__class__.__name__.lower()
        is_linear_like = (
            isinstance(base_mod, nn.Linear)
            or "conv1d" in cls_name
            or hasattr(base_mod, "nf")
        )
        if not is_linear_like:
            return False
        if isinstance(base_mod, SVDLinear):
            return False

        try:
            if not should_compress_linear(self.model, name):
                return False
        except Exception:
            pass

        if self._is_gqa_asymmetric(name, module):
            return False

        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    # ------------------------------------------------------------------
    # Calibration batch normalization (SW-FIX-02)
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_calibration_batches(calib_inputs: Any) -> list[Any]:
        """
        Convert whatever `get_calibration_tensors` returns into a list of batches.

        SW-FIX-02: A single tuple like `(input_ids, attention_mask)` is treated
        as ONE batch, not as two batches. Detection rule: a batch is a tuple/
        list ONLY IF its elements are themselves tensors/dicts AND the tuple
        length is small (≤ 3). Otherwise, the top-level container is a list of
        batches.
        """
        if calib_inputs is None:
            return []

        # Single tensor / dict / small tuple → single batch
        if isinstance(calib_inputs, torch.Tensor):
            return [calib_inputs]
        if isinstance(calib_inputs, dict):
            return [calib_inputs]

        if isinstance(calib_inputs, (list, tuple)):
            if len(calib_inputs) == 0:
                return []

            # Case A: top-level list/tuple where every element is a batch
            # (dict, tensor, or tuple-of-tensors)
            all_batches = all(
                isinstance(x, (dict, torch.Tensor, list, tuple))
                for x in calib_inputs
            )
            if all_batches:
                # Distinguish "list of batches" from "single batch as tuple".
                # Heuristic: single batch if the outer tuple contains only
                # tensors (e.g., (input_ids, attention_mask)).
                if all(isinstance(x, torch.Tensor) for x in calib_inputs) \
                        and len(calib_inputs) <= 3:
                    return [calib_inputs]
                # Otherwise, it's a list of batches.
                return list(calib_inputs)

        return [calib_inputs]

    # ------------------------------------------------------------------
    # Activation collection (SW-FIX-04, SW-FIX-07, SW-FIX-16)
    # ------------------------------------------------------------------
    def _collect_activations(self) -> dict[str, tuple[torch.Tensor, int]]:
        """
        Collect activations and accumulate cross-covariance C = E[Y^T X] per layer.

        Returns {name: (cross_cov_fp32, n_tokens)}.
        cross_cov shape: [out_features, in_features] in fp32 on CPU.
        """
        accum: dict[str, dict[str, Any]] = {}
        hooks = []
        compute_dtype = torch.float64 if self.use_fp64_cross_cov else torch.float32

        def make_hook(layer_name: str):
            def hook(module: nn.Module, inp: tuple[Any, ...], outp: Any) -> None:
                if not inp or inp[0] is None:
                    return
                x_in = inp[0].detach()
                if isinstance(outp, torch.Tensor):
                    x_out = outp.detach()
                elif (isinstance(outp, (tuple, list))
                      and len(outp) > 0
                      and isinstance(outp[0], torch.Tensor)):
                    x_out = outp[0].detach()
                else:
                    return

                in_f = x_in.shape[-1]
                out_f = x_out.shape[-1]

                x_in_2d = x_in.reshape(-1, in_f).float()
                x_out_2d = x_out.reshape(-1, out_f).float()

                n = x_in_2d.shape[0]
                if n > self.max_calibration_tokens:
                    x_in_2d = x_in_2d[: self.max_calibration_tokens]
                    x_out_2d = x_out_2d[: self.max_calibration_tokens]
                    n = self.max_calibration_tokens

                if n == 0:
                    return

                if layer_name not in accum:
                    accum[layer_name] = {
                        "cov": torch.zeros(out_f, in_f, dtype=compute_dtype),
                        "n": 0,
                        "in_f": in_f,
                        "out_f": out_f,
                    }
                elif accum[layer_name]["in_f"] != in_f or accum[layer_name]["out_f"] != out_f:
                    # Shape changed across batches → skip
                    return

                with torch.no_grad():
                    cov_local = (
                        x_out_2d.to(compute_dtype).t() @ x_in_2d.to(compute_dtype)
                    ).cpu()
                accum[layer_name]["cov"] += cov_local
                accum[layer_name]["n"] += n
            return hook

        target_layers = [
            (name, mod) for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]
        if not target_layers:
            return {}

        for name, mod in target_layers:
            hooks.append(mod.register_forward_hook(make_hook(name)))

        device = next(self.model.parameters(), torch.tensor(0.0)).device
        self.model.eval()

        try:
            calib_inputs = get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=device,
            )
            batches = self._normalize_calibration_batches(calib_inputs)
            n_batches = min(len(batches), self.num_calibration_batches)

            if n_batches == 0:
                logger.warning("SwiftSVDActivation: no calibration batches available.")
                return {}

            with torch.no_grad():
                for i in range(n_batches):
                    batch = batches[i]
                    try:
                        if isinstance(batch, dict):
                            b_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                                     for k, v in batch.items()}
                            self.model(**b_dev)
                        elif isinstance(batch, torch.Tensor):
                            self.model(batch.to(device))
                        elif isinstance(batch, (list, tuple)):
                            b_dev = [v.to(device) if isinstance(v, torch.Tensor) else v
                                     for v in batch]
                            self.model(*b_dev)
                    except Exception as e:
                        logger.debug("Calibration batch %d forward failed: %s", i, e)
        finally:
            for h in hooks:
                h.remove()

        aggregated: dict[str, tuple[torch.Tensor, int]] = {}
        for name, data in accum.items():
            n = max(1, data["n"])
            cov = (data["cov"] / n).float()
            aggregated[name] = (cov, int(n))
        return aggregated

    # ------------------------------------------------------------------
    # LoRA rank detection (SW-FIX-03)
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_lora_rank(module: nn.Module) -> int:
        for attr in ("r", "lora_r", "rank"):
            val = getattr(module, attr, None)
            if isinstance(val, int) and val > 0:
                return val
        la = getattr(module, "lora_A", None)
        if la is not None:
            if hasattr(la, "out_features"):
                try:
                    return int(la.out_features)
                except Exception:
                    pass
            if hasattr(la, "weight") and la.weight is not None and la.weight.dim() >= 1:
                return int(la.weight.shape[0])
            if isinstance(la, torch.Tensor) and la.dim() >= 1:
                return int(la.shape[0])
        return 0

    # ------------------------------------------------------------------
    # Rank selection (SW-FIX-01, SW-FIX-13)
    # ------------------------------------------------------------------
    def _select_rank(self, in_f: int, out_f: int) -> int:
        """
        SW-FIX-13: clear, monotonic rank selection.

        1. max_r = min(in_f, out_f)
        2. Dense break-even rank (where param count of SVD == param count of dense):
             2r(in + out) < in · out  →  r < in·out / (2(in+out))
           We use `param_cap = (in·out) // (in + out) - 1` (a slightly stricter bound
           often used in the codebase) and clamp to max_r.
        3. requested_r = clamp(self.rank, self.min_rank, param_cap)
        """
        max_r = min(in_f, out_f)
        if in_f * out_f > 16:
            param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        else:
            param_cap = max_r
        param_cap = max(1, min(param_cap, max_r))

        r_lower = min(self.min_rank, max_r, param_cap)
        r_upper = min(self.rank, max_r, param_cap)

        # SW-FIX-13: monotone selection — start from r_lower, aim for r_upper.
        r = max(r_lower, r_upper)
        r = max(1, min(r, param_cap))
        return r

    # ------------------------------------------------------------------
    # Per-layer compression
    # ------------------------------------------------------------------
    def compress_linear(
        self,
        layer: nn.Module,
        act_pair: tuple[torch.Tensor, int] | None = None,
    ) -> SVDLinear | None:
        base_layer = getattr(layer, "base_layer", layer)
        raw_w = base_layer.weight.data
        cls_name = base_layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(base_layer, "nf")

        W = raw_w.t().contiguous().clone().float() if is_conv1d else raw_w.clone().float()
        out_f, in_f = W.shape

        r = self._select_rank(in_f, out_f)
        device = raw_w.device
        dtype = raw_w.dtype
        A: torch.Tensor | None = None
        B: torch.Tensor | None = None

        # ---- Activation subspace projection ----
        if act_pair is not None:
            try:
                cov, _n = act_pair
                if cov.shape[0] == out_f and cov.shape[1] == in_f:
                    C = cov.to(device=device, dtype=torch.float32)
                    U_c, _, Vt_c = torch.linalg.svd(C, full_matrices=False)
                    r_act = min(r, U_c.shape[1], Vt_c.shape[0])
                    if r_act > 0:
                        U_act = U_c[:, :r_act]
                        V_act = Vt_c[:r_act, :].t().contiguous()

                        # M = U_act^T W V_act
                        M = U_act.t().contiguous() @ W @ V_act
                        U_m, S_m, Vt_m = torch.linalg.svd(M, full_matrices=False)
                        r_act = min(r_act, len(S_m))
                        if r_act > 0:
                            sqrt_S_m = torch.sqrt(torch.clamp(S_m[:r_act], min=1e-12))
                            A = (U_act @ U_m[:, :r_act]) * sqrt_S_m.unsqueeze(0)
                            B = sqrt_S_m.unsqueeze(1) @ (Vt_m[:r_act, :] @ V_act.t().contiguous())
                            r = r_act
            except Exception as e:
                logger.debug("Activation subspace projection failed (%s). Falling back to direct SVD.", e)
                A, B = None, None

        # ---- Direct SVD fallback ----
        if A is None or B is None:
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r = min(r, len(S))
                sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
                A = U[:, :r] * sqrt_S.unsqueeze(0)
                B = sqrt_S.unsqueeze(1) @ Vt[:r, :]
            except Exception as e:
                logger.warning("SwiftSVDActivation SVD failure: %s", e)
                return None

        # ---- Error check ----
        W_rec = A @ B
        norm_orig = torch.norm(W, p="fro").item()
        rel_err = (torch.norm(W - W_rec) / max(norm_orig, 1e-12)).item()

        if not math.isfinite(rel_err) or rel_err > self.max_error:
            # Try direct SVD (may have better conditioning)
            try:
                U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                r_d = min(r, len(S))
                sqrt_S = torch.sqrt(torch.clamp(S[:r_d], min=1e-12))
                A_d = U[:, :r_d] * sqrt_S.unsqueeze(0)
                B_d = sqrt_S.unsqueeze(1) @ Vt[:r_d, :]
                W_rec_d = A_d @ B_d
                err_d = (torch.norm(W - W_rec_d) / max(norm_orig, 1e-12)).item()
                if err_d < rel_err:
                    A, B, r, rel_err = A_d, B_d, r_d, err_d
            except Exception as e:
                logger.debug("Direct SVD fallback failed: %s", e)

        # SW-FIX-14: preserve original on high error
        if not math.isfinite(rel_err) or rel_err > self.max_error:
            logger.warning(
                "SwiftSVDActivation reconstruction error %.4f exceeded threshold %.4f on %s.",
                rel_err, self.max_error, getattr(base_layer, "_name", "?"),
            )
            return None

        # ---- Build SVDLinear ----
        effective_lora_r = self.lora_r
        detected = self._detect_lora_rank(layer)
        if detected > 0:
            effective_lora_r = detected

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=base_layer.bias is not None,
            dtype=dtype,
            device=device,
            lora_r=effective_lora_r,
            lora_alpha=self.lora_alpha,
        )
        svd_layer.set_low_rank_factors(
            U_factor=A.to(dtype=dtype, device=device),
            V_factor=B.to(dtype=dtype, device=device),
        )
        if base_layer.bias is not None and svd_layer.bias is not None:
            svd_layer.bias.data.copy_(base_layer.bias.data)

        # ---- LoRA transfer ----
        if effective_lora_r > 0 and getattr(svd_layer, "use_lora", False):
            try:
                src_lA = getattr(layer, "lora_A", None)
                src_lB = getattr(layer, "lora_B", None)
                if src_lA is not None and src_lB is not None:
                    src_lA_w = getattr(src_lA, "weight", src_lA)
                    src_lB_w = getattr(src_lB, "weight", src_lB)
                    dst_lA_w = getattr(svd_layer.lora_A, "weight", svd_layer.lora_A)
                    dst_lB_w = getattr(svd_layer.lora_B, "weight", svd_layer.lora_B)
                    if src_lA_w.shape == dst_lA_w.shape:
                        dst_lA_w.data.copy_(src_lA_w.data)
                    if src_lB_w.shape == dst_lB_w.shape:
                        dst_lB_w.data.copy_(src_lB_w.data)
            except Exception as e:
                logger.debug("LoRA transfer failed: %s", e)

        svd_layer.reconstruction_error = rel_err
        return svd_layer

    # ------------------------------------------------------------------
    # Weight-sharing detection
    # ------------------------------------------------------------------
    @staticmethod
    def _get_weight_tensor(module: nn.Module) -> torch.Tensor | None:
        base = getattr(module, "base_layer", module)
        return getattr(base, "weight", None)

    # ------------------------------------------------------------------
    # Main compress
    # ------------------------------------------------------------------
    def compress(self, **kwargs: Any) -> tuple[nn.Module, dict[str, Any]]:
        self.compressed_layers = []
        self.fallback_layers = []
        self.skipped_layers = []
        self.layer_errors = {}
        self.replaced_layers = []
        self._orig_storage_bytes = 0
        self._comp_storage_bytes = 0
        seen_weights: set[int] = set()

        target_layers: list[tuple[str, nn.Module]] = []
        for name, mod in list(self.model.named_modules()):
            if not self._is_target_module(name, mod):
                continue
            w = self._get_weight_tensor(mod)
            if w is None:
                continue
            w_id = id(w)
            if w_id in seen_weights:
                self.skipped_layers.append(name)
                continue
            seen_weights.add(w_id)
            target_layers.append((name, mod))

        if not target_layers:
            logger.warning("SwiftSVDActivation: no compressible linear layers found.")
            return self.model, self.get_compression_stats()

        activations = self._collect_activations()

        for name, module in target_layers:
            act_pair = activations.get(name, None)
            base_layer = getattr(module, "base_layer", module)
            raw_w = base_layer.weight.data

            # Original bytes
            orig_bytes = raw_w.numel() * raw_w.element_size()
            if getattr(base_layer, "bias", None) is not None:
                orig_bytes += base_layer.bias.numel() * base_layer.bias.element_size()
            self._orig_storage_bytes += orig_bytes

            svd_layer = self.compress_linear(module, act_pair=act_pair)
            if svd_layer is not None:
                err = getattr(svd_layer, "reconstruction_error", 0.0)
                self.layer_errors[name] = err
                self.replaced_layers.append({
                    "name": name,
                    "rank": svd_layer.rank,
                    "error": err,
                })

                if hasattr(module, "base_layer"):
                    module.base_layer = svd_layer
                    for attr in ("lora_A", "lora_B"):
                        sub = getattr(module, attr, None)
                        if sub is None:
                            continue
                        params = sub.parameters() if hasattr(sub, "parameters") else iter([sub])
                        for p in params:
                            if hasattr(p, "requires_grad"):
                                p.requires_grad = True
                else:
                    replace_module(self.model, name, svd_layer)

                self.compressed_layers.append(name)
                # Honest byte count
                comp_bytes = 0
                for p in svd_layer.parameters():
                    comp_bytes += p.numel() * p.element_size()
                for b in svd_layer.buffers():
                    if b is not None:
                        comp_bytes += b.numel() * b.element_size()
                self._comp_storage_bytes += comp_bytes
            else:
                self.fallback_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False
                # Original storage retained
                self._comp_storage_bytes += orig_bytes

        # ---- Forward sanity check (SW-FIX-15) ----
        forward_check_success = True
        try:
            with torch.no_grad():
                dev = next(self.model.parameters(), torch.tensor(0.0)).device
                cfg = getattr(self.model, "config", None)
                vocab = getattr(cfg, "vocab_size", None) if cfg else None
                if vocab is None:
                    vocab = 100
                dummy = torch.randint(0, min(vocab, 100), (1, 8), device=dev)
                try:
                    self.model(input_ids=dummy,
                                attention_mask=torch.ones_like(dummy))
                except Exception:
                    try:
                        self.model(dummy)
                    except Exception as e2:
                        logger.warning("Post-compression forward check failed: %s", e2)
                        forward_check_success = False
        except Exception as e:
            logger.warning("Post-compression forward check warning: %s", e)
            forward_check_success = False

        try:
            self._compressed_params = self._count_params(self.model)
        except Exception:
            self._compressed_params = sum(p.numel() for p in self.model.parameters())

        avg_err = (
            sum(self.layer_errors[k] for k in self.compressed_layers)
            / max(1, len(self.compressed_layers))
            if self.compressed_layers else 0.0
        )
        try:
            stats = self.get_compression_stats()
        except Exception:
            stats = {}

        storage_ratio = (
            self._orig_storage_bytes / self._comp_storage_bytes
            if self._comp_storage_bytes > 0 else 1.0
        )
        effective_ratio = (
            self._comp_storage_bytes / self._orig_storage_bytes
            if self._orig_storage_bytes > 0 else 1.0
        )

        stats.update({
            "compressed_layers": self.compressed_layers,
            "fallback_layers": self.fallback_layers,
            "skipped_layers": self.skipped_layers,
            "num_compressed": len(self.compressed_layers),
            "num_fallback": len(self.fallback_layers),
            "num_skipped": len(self.skipped_layers),
            "layer_errors": self.layer_errors,
            "avg_reconstruction_error": avg_err,
            "max_error": self.max_error,
            "fallback_to_original": self.fallback_to_original,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            # SW-FIX-10: honest accounting
            "orig_storage_bytes": self._orig_storage_bytes,
            "comp_storage_bytes": self._comp_storage_bytes,
            "storage_compression_ratio": round(storage_ratio, 3),
            "storage_reduction_pct": round(100.0 * (1.0 - effective_ratio), 2),
            "effective_parameter_ratio": round(effective_ratio, 4),
            "forward_check_success": forward_check_success,
            "method": "swift_svd_act",
        })
        self.stats = stats
        logger.info(
            "Swift-SVD-ACT: %d compressed, %d fallback, avg_err=%.4f, storage_ratio=%.2fx",
            len(self.compressed_layers), len(self.fallback_layers), avg_err, storage_ratio,
        )
        return self.model, stats


# ===========================================================================
# Factory
# ===========================================================================
@COMPRESSOR_REGISTRY.register("swift_svd_act_factory")
def swift_svd_act_factory(
    model: nn.Module, comp_config: Any = None, **kwargs: Any
) -> Any:
    """Swift-SVD activation-aware factory function."""
    rank = kwargs.pop("rank", _cfg_get(comp_config, "swift_svd_act_rank",
                                       _cfg_get(comp_config, "rank", 64)))
    max_error = kwargs.pop("max_error", _cfg_get(comp_config, "max_error", 0.50))
    fallback = kwargs.pop("fallback_to_original",
                          _cfg_get(comp_config, "fallback_to_original", True))
    lora_r = kwargs.pop("lora_r", _cfg_get(comp_config, "lora_r", 0))
    lora_alpha = kwargs.pop("lora_alpha", _cfg_get(comp_config, "lora_alpha", 1.0))
    target_attn = kwargs.pop("target_attention_only",
                             _cfg_get(comp_config, "target_attention_only", False))
    num_calib = kwargs.pop("num_calibration_batches",
                           _cfg_get(comp_config, "num_calibration_batches", 8))
    max_tokens = kwargs.pop("max_calibration_tokens",
                            _cfg_get(comp_config, "max_calibration_tokens", 32768))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = SwiftSVDActivationCompressor(
        model,
        rank=rank,
        max_error=max_error,
        fallback_to_original=fallback,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_attention_only=target_attn,
        num_calibration_batches=num_calib,
        max_calibration_tokens=max_tokens,
        comp_config=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model


__all__ = [
    "SwiftSVDActivationCompressor",
    "swift_svd_act_factory",
]
