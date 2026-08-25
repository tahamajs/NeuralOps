"""
Bonsai: Iterative Magnitude and Taylor Pruning (IMP) with Gradient Descent Recovery & [CLS] Routing Protection.
Registered in COMPRESSOR_REGISTRY as 'bonsai' and 'bonsai_pruner'.

Mathematical Formulation:
1. Polynomial Sparsity Schedule:
   $$s_t = s_{\\text{target}} \\cdot \\left(1 - \\left(1 - \\frac{t}{T}\\right)^3\\right), \\quad t \\in \\{1, \\dots, T\\}$$
2. Protected [CLS] / Router Modules:
   $$\\mathcal{M}_{\\text{router}} = \\mathbf{1}, \\quad \\mathcal{M}_{\\text{classifier}} = \\mathbf{1}$$
3. Dynamic Saliency & Mask Update:
   $$S_t(W_{ij}) = |W_{ij}^{(t)}| \\cdot \\mathbb{E}[|x_j|], \\quad \\mathcal{M}^{(t)} = \\Pi_{\\mathcal{S}}(W^{(t)}, S_t, s_t)$$
4. Inter-Step Gradient Descent Reconstruction Recovery:
   $$\\min_W \\frac{1}{2} \\| (W \\odot \\mathcal{M}^{(t)}) X - W_0 X \\|_F^2 + \\frac{\\lambda}{2} \\|W - W_0\\|_F^2$$
   $$W^{(t)} \\leftarrow W^{(t)} - \\eta \\left( ((W^{(t)} \\odot \\mathcal{M}^{(t)}) X - W_0 X) X^\\top \\odot \\mathcal{M}^{(t)} \\right)$$
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import math
import torch
import torch.nn as nn

from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.pruning_utils import (
    apply_24_structured_mask,
    apply_block_mask,
    compute_taylor_first_order_saliency,
    get_calibration_tensors,
    recalibrate_layernorms,
)

logger = logging.getLogger(__name__)


class BonsaiPruner(BaseCompressor):
    """
    Bonsai Pruner: Progressive multi-step iterative pruning with cubic schedule,
    inter-step gradient descent recovery, and [CLS] routing protection.
    """

    def __init__(
        self,
        model: nn.Module,
        sparsity: float = 0.5,
        steps: int = 8,
        recovery_steps: int = 3,
        recovery_lr: float = 1e-3,
        structured_24: bool = False,
        block_size: Tuple[int, int] = (4, 4),
        protect_cls: bool = True,
        recalibrate_ln: bool = True,
        tokenizer: Optional[Any] = None,
        calibration_loader: Optional[Any] = None,
        calibration_texts: Optional[List[str]] = None,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="bonsai", **kwargs)
        self.sparsity = max(0.0, min(float(sparsity), 0.95))
        self.steps = max(1, int(steps))
        self.recovery_steps = max(0, int(recovery_steps))
        self.recovery_lr = float(recovery_lr)
        self.structured_24 = structured_24
        self.block_size = block_size
        self.protect_cls = protect_cls
        self.recalibrate_ln = recalibrate_ln
        self.tokenizer = tokenizer or getattr(model, "tokenizer", None)
        self.calibration_loader = calibration_loader
        self.calibration_texts = calibration_texts
        self.target_device = (
            torch.device(device)
            if device is not None
            else (next(model.parameters(), torch.tensor(0.0)).device)
        )
        self.stats: Dict[str, Any] = {}

    def _is_protected_layer(self, name: str) -> bool:
        """Determines if a layer is a protected [CLS], router, or classification head module."""
        if not self.protect_cls:
            return False
        name_lower = name.lower()
        protected_keywords = ["router", "gate", "cls", "classifier", "head", "embed", "score", "lm_head"]
        return any(kw in name_lower for kw in protected_keywords)

    def _collect_activations(self) -> Dict[str, torch.Tensor]:
        """Capture activations across linear layers."""
        self.model.eval()
        activations: Dict[str, List[torch.Tensor]] = {}
        hooks = []

        def make_hook(name: str):
            def hook(module: nn.Module, inp: Tuple[torch.Tensor, ...], outp: Any):
                if inp and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach()
                    if name not in activations:
                        activations[name] = []
                    activations[name].append(x.cpu())
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        calib_inputs = get_calibration_tensors(
            self.model,
            tokenizer=self.tokenizer,
            calibration_loader=self.calibration_loader,
            calibration_texts=self.calibration_texts,
            device=self.target_device,
        )

        with torch.no_grad():
            try:
                if isinstance(calib_inputs, dict):
                    self.model(**calib_inputs)
                elif isinstance(calib_inputs, torch.Tensor):
                    self.model(calib_inputs)
                elif isinstance(calib_inputs, (list, tuple)):
                    self.model(*calib_inputs)
            except Exception as e:
                logger.debug("Bonsai calibration exception: %s", e)

        for h in hooks:
            h.remove()

        aggregated: Dict[str, torch.Tensor] = {}
        for name, act_list in activations.items():
            if act_list:
                aggregated[name] = torch.cat([a.reshape(-1, a.shape[-1]) for a in act_list], dim=0)

        return aggregated

    def compress(self, **kwargs: Any) -> Tuple[nn.Module, Dict[str, Any]]:
        """Executes progressive iterative pruning with inter-step gradient descent recovery."""
        activations_map = self._collect_activations()
        pruned_layers_count = 0
        protected_layers_count = 0

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                if self._is_protected_layer(name):
                    protected_layers_count += 1
                    continue

                W0 = module.weight.data.clone().float()
                device = W0.device
                dtype = module.weight.dtype
                W = W0.clone()

                act = activations_map.get(name, None)
                X = None
                if act is not None and act.numel() > 0:
                    X = act.to(device).float()
                    if X.shape[-1] != W.shape[1]:
                        X = None

                final_mask = torch.ones_like(W)

                # Iterative Pruning Loop
                for s in range(self.steps):
                    t_ratio = (s + 1.0) / self.steps
                    current_sp = self.sparsity * (1.0 - (1.0 - t_ratio) ** 3)
                    saliency = compute_taylor_first_order_saliency(W, activations=X)

                    # 1. Compute Mask for step s
                    if self.structured_24:
                        W_pruned, final_mask = apply_24_structured_mask(W, saliency=saliency, dim=1)
                    elif self.block_size != (1, 1):
                        W_pruned, final_mask = apply_block_mask(
                            W, block_size=self.block_size, sparsity=current_sp, saliency=saliency
                        )
                    else:
                        flat = saliency.reshape(-1)
                        k = max(1, int(flat.numel() * (1.0 - current_sp)))
                        thr = torch.topk(flat, k, largest=True).values[-1]
                        final_mask = (saliency >= thr).float()
                        W_pruned = W * final_mask

                    W = W_pruned

                    # 2. Gradient Descent Recovery between pruning steps
                    if self.recovery_steps > 0 and X is not None:
                        # Normalize covariance X^T X
                        cov_X = torch.matmul(X.T, X) / max(X.shape[0], 1)  # [d_in, d_in]
                        # Target output: W0 @ cov_X
                        W0_cov = torch.matmul(W0, cov_X)

                        step_size = self.recovery_lr / (torch.norm(cov_X).item() + 1e-6)
                        for _ in range(self.recovery_steps):
                            # grad = ((W @ cov_X) - W0_cov) * mask
                            curr_cov = torch.matmul(W, cov_X)
                            grad = (curr_cov - W0_cov) * final_mask
                            W = W - step_size * grad
                            W = W * final_mask  # ensure pruned coordinates stay zero

                module.weight.data = W.to(dtype=dtype, device=device)
                pruned_layers_count += 1

        # LayerNorm Recalibration
        if self.recalibrate_ln:
            recalibrate_layernorms(
                self.model,
                tokenizer=self.tokenizer,
                calibration_loader=self.calibration_loader,
                calibration_texts=self.calibration_texts,
                device=self.target_device,
            )

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update({
            "pruned_layers": pruned_layers_count,
            "protected_layers": protected_layers_count,
            "steps": self.steps,
            "recovery_steps": self.recovery_steps,
            "sparsity_target": self.sparsity,
            "structured_24": self.structured_24,
            "protect_cls": self.protect_cls,
            "recalibrate_ln": self.recalibrate_ln,
            "method": "bonsai_iterative_pruning_with_gd_recovery",
        })
        self.stats = stats
        return self.model, stats


@COMPRESSOR_REGISTRY.register("bonsai")
@COMPRESSOR_REGISTRY.register("bonsai_pruner")
def bonsai_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """Factory function for Bonsai pruner."""
    sparsity = getattr(comp_config, "bonsai_sparsity", getattr(comp_config, "sparsity", 0.5)) if comp_config else kwargs.get("sparsity", 0.5)
    steps = getattr(comp_config, "bonsai_steps", getattr(comp_config, "steps", 8)) if comp_config else kwargs.get("steps", 8)
    recovery_steps = getattr(comp_config, "bonsai_recovery_steps", getattr(comp_config, "recovery_steps", 3)) if comp_config else kwargs.get("recovery_steps", 3)
    structured_24 = getattr(comp_config, "bonsai_structured_24", getattr(comp_config, "structured_24", False)) if comp_config else kwargs.get("structured_24", False)
    protect_cls = getattr(comp_config, "bonsai_protect_cls", kwargs.get("protect_cls", True))
    recalibrate_ln = getattr(comp_config, "bonsai_recalibrate_ln", kwargs.get("recalibrate_ln", True))

    compressor = BonsaiPruner(
        model,
        sparsity=sparsity,
        steps=steps,
        recovery_steps=recovery_steps,
        structured_24=structured_24,
        protect_cls=protect_cls,
        recalibrate_ln=recalibrate_ln,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model

__all__ = [
    "BonsaiPruner",
    "bonsai_factory",
]
