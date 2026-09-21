r"""
SliceGPT PCA Compressor: Principal Component Analysis based Dimensionality Slicing.
Registered in COMPRESSOR_REGISTRY as 'slicegpt_pca' and 'slicegpt_pca_factory'.

Mathematical Formulation:
1. Activation Sample Matrix $X \in \mathbb{R}^{N 	imes d_{in}}$
2. Empirical Covariance: $\Sigma = rac{1}{N - 1} (X - \mu)^	op (X - \mu)$
3. Eigendecomposition: $\Sigma = V \Lambda V^	op$
4. Retain principal coordinates covering $(1 - 	ext{sparsity})$ cumulative variance:
   $$k = \min \left\{ r \;\middle|\; rac{\sum_{i=1}^r \lambda_i}{\sum_{i=1}^d \lambda_i} \ge 1 - 	ext{sparsity} 
ight\}$$
5. Transform weights: $W \leftarrow (W V) P_k P_k^	op V^	op$
6. Optional LoRA Adapter Integration:
   $$y = (W_{	ext{sliced}} x + b) + rac{lpha}{r} (B A x)$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple, Union
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import get_calibration_tensors
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class SliceGPTPCALoRALinear(nn.Module):
    """
    Linear layer containing sliced/PCA-projected base weights (frozen)
    with trainable LoRA adapters (lora_A, lora_B) for post-compression fine-tuning.
    Base weights are frozen with requires_grad=False; initial LoRA contribution is zero.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
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
            self.register_parameter('bias', None)

        if self.lora_r > 0:
            self.lora_A = nn.Parameter(torch.empty((self.lora_r, in_features), device=target_device, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros((out_features, self.lora_r), device=target_device, dtype=torch.float32))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.register_parameter('lora_A', None)
            self.register_parameter('lora_B', None)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        sliced_weight: torch.Tensor,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
    ) -> SliceGPTPCALoRALinear:
        base_mod = getattr(linear, 'base_layer', linear)
        bias = base_mod.bias.detach() if hasattr(base_mod, 'bias') and base_mod.bias is not None else None
        in_features = getattr(base_mod, 'in_features', sliced_weight.shape[1])
        out_features = getattr(base_mod, 'out_features', sliced_weight.shape[0])
        
        has_existing_lora = hasattr(linear, 'lora_A') and hasattr(linear, 'lora_B')
        effective_lora_r = lora_r
        if has_existing_lora and getattr(linear, 'r', 0) > 0:
            effective_lora_r = getattr(linear, 'r', lora_r)

        layer = cls(
            in_features=in_features,
            out_features=out_features,
            weight=sliced_weight,
            bias=bias,
            lora_r=effective_lora_r,
            lora_alpha=lora_alpha,
            device=sliced_weight.device,
            dtype=sliced_weight.dtype,
        )

        if has_existing_lora and layer.lora_r > 0:
            try:
                if hasattr(linear.lora_A, 'weight') and layer.lora_A is not None:
                    layer.lora_A.data.copy_(linear.lora_A.weight.data)
                if hasattr(linear.lora_B, 'weight') and layer.lora_B is not None:
                    layer.lora_B.data.copy_(linear.lora_B.weight.data)
            except Exception as e_lora:
                logger.warning('Could not copy existing LoRA adapters in SliceGPTPCA: %s', e_lora)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)
        if self.lora_r > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_delta = F.linear(
                F.linear(x.float(), self.lora_A),
                self.lora_B,
            ) * self.scaling
            out = out + lora_delta.to(out.dtype)
        return out


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@COMPRESSOR_REGISTRY.register('slicegpt_pca')
class SliceGPTPCACompressor(BaseCompressor):
    """
    Production-hardened SliceGPT PCA-based compressor that performs activation-aware
    principal coordinate projection and dimension slicing on Linear and Conv1D layers.
    """

    def __init__(
        self,
        model: nn.Module,
        calibration_loader: Optional[Any] = None,
        sparsity: float = 0.3,
        tokenizer: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        skip_attention_qkv: bool = False,
        target_attention_only: bool = False,
        use_lora: bool = False,
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        max_error: float = 0.50,
        fallback_to_original: bool = True,
        min_numel: int = 16,
        target_modules: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name='slicegpt_pca', **kwargs)
        cfg_obj = cfg or kwargs.get('comp_config', None)

        self.calibration_loader = calibration_loader
        self.sparsity = max(0.0, min(float(sparsity or _cfg_get(cfg_obj, 'slicegpt_pca_sparsity', _cfg_get(cfg_obj, 'sparsity', 0.3))), 0.95))
        self.tokenizer = tokenizer or getattr(model, 'tokenizer', None) or _cfg_get(cfg_obj, 'tokenizer', None)
        self.calibration_texts = calibration_texts or _cfg_get(cfg_obj, 'calibration_texts', None)
        self.skip_attention_qkv = bool(skip_attention_qkv if skip_attention_qkv is not None else _cfg_get(cfg_obj, 'skip_attention_qkv', False))
        self.target_attention_only = bool(target_attention_only if target_attention_only is not None else _cfg_get(cfg_obj, 'target_attention_only', False))
        self.use_lora = bool(use_lora or _cfg_get(cfg_obj, 'use_lora', False) or (lora_r > 0))
        self.lora_r = int(lora_r or _cfg_get(cfg_obj, 'lora_r', 0))
        self.lora_alpha = float(lora_alpha or _cfg_get(cfg_obj, 'lora_alpha', 1.0))
        self.max_error = float(max_error or _cfg_get(cfg_obj, 'max_error', 0.50))
        self.fallback_to_original = bool(fallback_to_original if fallback_to_original is not None else _cfg_get(cfg_obj, 'fallback_to_original', True))
        self.min_numel = max(1, int(min_numel or _cfg_get(cfg_obj, 'min_numel', 16)))
        self.target_modules = target_modules

        # Target device derivation
        if device is not None:
            try:
                self.target_device = torch.device(device)
            except Exception:
                self.target_device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        else:
            self.target_device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')

        # Diagnostics & Tracking
        self.visited_modules: Set[int] = set()
        self.sliced_layers: List[str] = []
        self.protected_layers: List[str] = []
        self.skipped_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.failed_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.stats: Dict[str, Any] = {}

    def _is_gqa_asymmetric(self, name: str, module: nn.Module) -> bool:
        """Check if layer belongs to an attention projection in a GQA architecture."""
        cfg = getattr(self.model, 'config', None)
        num_heads = getattr(cfg, 'num_attention_heads', None)
        num_kv_heads = getattr(cfg, 'num_key_value_heads', None)
        is_gqa = (num_heads is not None and num_kv_heads is not None and num_kv_heads < num_heads)
        
        name_l = name.lower()
        is_attn_qkv = any(p in name_l for p in [
            'q_proj', 'k_proj', 'v_proj', 'c_attn', 'query', 'key', 'value',
            'q_lin', 'k_lin', 'v_lin', 'self_attn.q', 'self_attn.k', 'self_attn.v'
        ])
        return (is_gqa or self.skip_attention_qkv) and is_attn_qkv

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        name_l = name.lower()
        # 1. Protection for critical / pooler / classification / embedding / CLS layers
        if any(k in name_l for k in [
            'classifier', 'score', 'lm_head', 'pooler', 'embed', 'embedding',
            'cls_head', 'head', 'logits_processor'
        ]):
            self.protected_layers.append(name)
            return False

        # Target attention only filter
        if self.target_attention_only:
            attn_keywords = ['attn', 'attention', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'c_attn', 'c_proj', 'query', 'key', 'value']
            if not any(k in name_l for k in attn_keywords):
                return False

        # 2. GQA-aware skipping
        if self._is_gqa_asymmetric(name, module):
            self.skipped_layers.append(name)
            return False

        # 3. Check linear or Conv1D nature
        base_mod = getattr(module, 'base_layer', module)
        cls_name = base_mod.__class__.__name__
        raw_w = getattr(base_mod, 'weight', None)
        if raw_w is None or not isinstance(raw_w, torch.Tensor):
            return False

        is_conv1d = 'conv1d' in cls_name.lower() or hasattr(base_mod, 'nf')
        is_linear_like = isinstance(base_mod, nn.Linear) or is_conv1d
        if not is_linear_like:
            return False

        # 4. Tiny layer guard
        if raw_w.numel() < self.min_numel:
            self.skipped_layers.append(name)
            return False

        # 5. Respect framework compatibility rules
        if not should_compress_linear(self.model, name):
            self.protected_layers.append(name)
            return False

        # 6. User-defined explicit target modules
        if self.target_modules is not None:
            return any(t in name for t in self.target_modules)
        return True

    def _collect_layer_activations(self) -> Dict[str, torch.Tensor]:
        """Capture streaming empirical covariance across target layers to eliminate memory spikes (H21)."""
        self.model.eval()
        sum_x: Dict[str, torch.Tensor] = {}
        sum_xx: Dict[str, torch.Tensor] = {}
        counts: Dict[str, int] = {}
        hooks = []

        def make_hook(layer_name: str):
            def hook(module: nn.Module, inputs: Tuple[torch.Tensor, ...], output: Any):
                if inputs and isinstance(inputs[0], torch.Tensor):
                    inp = inputs[0].detach()
                    x_flat = inp.reshape(-1, inp.shape[-1]).float()
                    n = x_flat.shape[0]
                    s_x = x_flat.sum(dim=0, keepdim=True).cpu()
                    s_xx = torch.matmul(x_flat.T, x_flat).cpu()
                    if layer_name not in sum_x:
                        sum_x[layer_name] = s_x
                        sum_xx[layer_name] = s_xx
                        counts[layer_name] = n
                    else:
                        sum_x[layer_name] += s_x
                        sum_xx[layer_name] += s_xx
                        counts[layer_name] += n
            return hook

        try:
            for name, module in self.model.named_modules():
                if self._is_target_module(name, module):
                    hooks.append(module.register_forward_hook(make_hook(name)))

            calib_inputs = get_calibration_tensors(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
            )

            with torch.no_grad():
                try:
                    if isinstance(calib_inputs, dict):
                        self.model(**{k: v.to(self.target_device) if isinstance(v, torch.Tensor) else v for k, v in calib_inputs.items()})
                    elif isinstance(calib_inputs, torch.Tensor):
                        self.model(calib_inputs.to(self.target_device))
                    elif isinstance(calib_inputs, (list, tuple)):
                        self.model(*[t.to(self.target_device) if isinstance(t, torch.Tensor) else t for t in calib_inputs])
                except Exception as ex_calib:
                    logger.debug('SliceGPTPCA primary calibration forward failed (%s); trying synthetic fallback.', ex_calib)
                    try:
                        dummy_ids = torch.randint(0, 100, (1, 16), device=self.target_device)
                        self.model(input_ids=dummy_ids)
                    except Exception as ex_synth:
                        logger.warning('SliceGPTPCA calibration forward pass skipped: %s', ex_synth)
        finally:
            for h in hooks:
                try:
                    h.remove()
                except Exception as e:
                    logger.warning(f"copy_weights failed: {e}")

        cov_map: Dict[str, torch.Tensor] = {}
        for name, n in counts.items():
            if n > 1:
                mu = sum_x[name] / float(n)
                cov = (sum_xx[name] - float(n) * torch.matmul(mu.T, mu)) / float(max(1, n - 1))
                cov_map[name] = cov
        return cov_map

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compresses the model using PCA-based principal subspace projection and slicing.
        Returns (compressed_model, stats).
        """
        self.layer_errors = {}
        self.visited_modules.clear()
        self.sliced_layers.clear()
        self.protected_layers.clear()
        self.skipped_layers.clear()
        self.fallback_layers.clear()
        self.failed_layers.clear()

        activations_map = self._collect_layer_activations()
        sliced_layers_count = 0
        pruned_features_count = 0

        target_items = [(n, m) for n, m in self.model.named_modules() if self._is_target_module(n, m)]

        for name, module in target_items:
            # Tied weight protection
            mod_id = id(module)
            if mod_id in self.visited_modules:
                logger.debug('SliceGPTPCA: Skipping tied module %s', name)
                self.skipped_layers.append(name)
                continue
            self.visited_modules.add(mod_id)

            try:
                base_mod = getattr(module, 'base_layer', module)
                raw_w = base_mod.weight.data
                cls_name = base_mod.__class__.__name__
                is_conv1d = 'conv1d' in cls_name.lower() or hasattr(base_mod, 'nf')
                device = raw_w.device
                dtype = raw_w.dtype

                # FP32 weight extraction
                if is_conv1d:
                    W = raw_w.t().contiguous().float()
                else:
                    W = raw_w.contiguous().float()
                d_out, d_in = W.shape

                cov_mat = activations_map.get(name, None)
                if cov_mat is not None and cov_mat.shape[0] == d_in and cov_mat.shape[1] == d_in:
                    cov = cov_mat.to(device=device, dtype=torch.float32) + 1e-5 * torch.eye(d_in, device=device)
                    try:
                        eigvals, V = torch.linalg.eigh(cov)
                        idx = torch.argsort(eigvals, descending=True)
                        eigvals = eigvals[idx]
                        V = V[:, idx]
                        eye_diff = torch.norm(torch.matmul(V.T, V) - torch.eye(d_in), p='fro').item()
                        if eye_diff > 1e-2 or torch.isnan(V).any():
                            raise ValueError(f'Eigh basis non-orthogonal (diff={eye_diff:.4f})')
                    except Exception as e_eigh:
                        logger.debug('SliceGPTPCA: eigh failed on %s (%s); fallback to QR.', name, e_eigh)
                        H = torch.randn(d_in, d_in)
                        V, _ = torch.linalg.qr(H)
                        eigvals = torch.ones(d_in)
                else:
                    H = torch.randn(d_in, d_in)
                    V, _ = torch.linalg.qr(H)
                    eigvals = torch.ones(d_in)

                V = V.to(device=device, dtype=torch.float32)
                W = W.to(device=device, dtype=torch.float32)

                # Compute cumulative variance
                pos_eigvals = torch.clamp(eigvals.to(device=device), min=0.0)
                total_var = torch.sum(pos_eigvals).item()
                if total_var > 0:
                    cum_var = torch.cumsum(pos_eigvals, dim=0) / total_var
                    target_var = 1.0 - self.sparsity
                    keep = int((cum_var < target_var).sum().item()) + 1
                    keep = max(1, min(keep, d_in))
                else:
                    keep = max(1, int(d_in * (1.0 - self.sparsity)))

                # Rotate weight into PCA coordinate space
                W_pca = torch.matmul(W, V)

                # Slice / zero out trailing PCA components
                if keep < d_in:
                    W_pca[:, keep:] = 0.0

                # Rotate back to original coordinate basis
                W_reconstructed = torch.matmul(W_pca, V.T)

                # Relative Frobenius error validation
                norm_orig = torch.norm(W, p='fro').clamp_min(1e-8)
                rel_err = (torch.norm(W - W_reconstructed, p='fro') / norm_orig).item()
                self.layer_errors[name] = float(rel_err)

                is_invalid = torch.isnan(W_reconstructed).any() or torch.isinf(W_reconstructed).any()
                if rel_err > self.max_error:
                    logger.warning(
                        'SliceGPTPCA: reconstruction error %.4f on %s exceeded threshold %.4f.',
                        rel_err, name, self.max_error
                    )
                    is_invalid = True

                if is_invalid and self.fallback_to_original:
                    logger.warning("Fallback triggered for %s. Layer is kept original and NOT wrapped in LoRA. It will not be fine-tuned.", name)
                    self.fallback_layers.append(name)
                    for p in module.parameters():
                        p.requires_grad = False
                    continue

                if keep < d_in:
                    pruned_features_count += (d_in - keep)

                # Commit compressed layer (with optional LoRA adapter)
                if self.use_lora and self.lora_r > 0:
                    lora_layer = SliceGPTPCALoRALinear.from_linear(
                        linear=module,
                        sliced_weight=W_reconstructed.to(dtype=dtype, device=device),
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                    )
                    replace_module(self.model, name, lora_layer)
                else:
                    final_w = W_reconstructed.t().contiguous() if is_conv1d else W_reconstructed.contiguous()
                    base_mod.weight.data.copy_(final_w.to(dtype=dtype, device=device))

                self.sliced_layers.append(name)
                sliced_layers_count += 1

            except Exception as e_layer:
                logger.warning('SliceGPTPCA: Failed to compress %s: %s', name, e_layer)
                self.failed_layers.append(name)
                for p in module.parameters():
                    p.requires_grad = False

        # Post-compression forward sanity check
        forward_check_success = True
        try:
            with torch.no_grad():
                dummy_input = torch.randint(0, 100, (1, 8), device=self.target_device)
                self.model(dummy_input)
        except Exception as ex_sanity:
            logger.warning('SliceGPTPCA post-compression forward sanity check warning: %s', ex_sanity)
            forward_check_success = False

        self._compressed_params = self._count_params(self.model)
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        stats = self.get_compression_stats()
        stats.update({
            'sliced_layers': self.sliced_layers,
            'sliced_layer_count': sliced_layers_count,
            'pruned_features': pruned_features_count,
            'protected_layers': self.protected_layers,
            'protected_layer_count': len(self.protected_layers),
            'skipped_layers': self.skipped_layers,
            'skipped_layer_count': len(self.skipped_layers),
            'fallback_layers': self.fallback_layers,
            'fallback_layer_count': len(self.fallback_layers),
            'failed_layers': self.failed_layers,
            'failed_layer_count': len(self.failed_layers),
            'sparsity_target': self.sparsity,
            'layer_errors': self.layer_errors,
            'avg_reconstruction_error': avg_err,
            'forward_check_success': forward_check_success,
            'skip_attention_qkv': self.skip_attention_qkv,
            'target_attention_only': self.target_attention_only,
            'use_lora': self.use_lora,
            'lora_r': self.lora_r,
            'method': 'slicegpt_pca',
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register('slicegpt_pca_factory')
def slicegpt_pca_factory(
    model: Any,
    comp_config: Any = None,
    return_stats: bool = False,
    return_tuple: bool = False,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """
    Factory function for SliceGPT PCA compressor.
    Returns (compressed_model, stats) if return_stats or return_tuple is True, else compressed_model.
    """
    sparsity = kwargs.pop('sparsity', _cfg_get(comp_config, 'slicegpt_pca_sparsity', _cfg_get(comp_config, 'sparsity', 0.3)))
    calibration_texts = kwargs.pop('calibration_texts', _cfg_get(comp_config, 'slicegpt_pca_calib_texts', _cfg_get(comp_config, 'calibration_texts', None)))
    tokenizer = kwargs.pop('tokenizer', getattr(model, 'tokenizer', None) or _cfg_get(comp_config, 'tokenizer', None))
    skip_attention_qkv = kwargs.pop('skip_attention_qkv', _cfg_get(comp_config, 'skip_attention_qkv', False))
    target_attention_only = kwargs.pop('target_attention_only', _cfg_get(comp_config, 'target_attention_only', False))
    use_lora = kwargs.pop('use_lora', _cfg_get(comp_config, 'use_lora', False))
    lora_r = kwargs.pop('lora_r', _cfg_get(comp_config, 'lora_r', 0))
    lora_alpha = kwargs.pop('lora_alpha', _cfg_get(comp_config, 'lora_alpha', 1.0))
    max_error = kwargs.pop('max_error', _cfg_get(comp_config, 'max_error', 0.50))
    fallback = kwargs.pop('fallback_to_original', _cfg_get(comp_config, 'fallback_to_original', True))

    compressor = SliceGPTPCACompressor(
        model,
        sparsity=sparsity,
        tokenizer=tokenizer,
        calibration_texts=calibration_texts,
        skip_attention_qkv=skip_attention_qkv,
        target_attention_only=target_attention_only,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats or return_tuple or kwargs.pop('return_both', False):
        return compressed_model, stats
    return compressed_model


__all__ = [
    'SliceGPTPCALoRALinear',
    'SliceGPTPCACompressor',
    'slicegpt_pca_factory',
]
