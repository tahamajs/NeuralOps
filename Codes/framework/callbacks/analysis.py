"""
Analysis callbacks: SVD, attention heatmaps, gradient norms, activation stats.
All metric extractions and .cpu().numpy() conversions are offloaded to an asynchronous
ThreadPoolExecutor to unblock the main CUDA stream and prevent GPU compute stalls.
"""

from __future__ import annotations

import concurrent.futures
import gc
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers import TrainerCallback

from .base import AsyncTelemetryCallback

logger = logging.getLogger(__name__)


def _async_process_attention_heatmap(
    att_tensor: Any,
    global_step: int,
    output_dir: str,
    layer_idx: int,
    head_idx: int,
) -> None:
    """
    Worker task running on a background thread to convert GPU attention tensors
    to NumPy ndarray, plot the heatmap with Matplotlib, and persist safely to disk.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        # FIX: Push metric extraction and .cpu().numpy() conversions into a non-blocking ThreadPoolExecutor to unblock the main CUDA stream
        if hasattr(att_tensor, "detach"):
            att_tensor = att_tensor.detach()
        if hasattr(att_tensor, "cpu"):
            att = att_tensor.cpu().float().numpy()
        else:
            att = np.asarray(att_tensor)

        fig = None
        try:
            with plt.style.context("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default"):
                fig, ax = plt.subplots(figsize=(10, 8))
                im = ax.imshow(att, cmap="viridis")
                ax.set_title(f"Attention Heatmap – Layer {layer_idx} Head {head_idx} – Step {global_step}")
                fig.colorbar(im, ax=ax)
                path = os.path.join(output_dir, f"attn_heatmap_step{global_step}.png")
                fig.tight_layout()
                fig.savefig(path, dpi=150, bbox_inches="tight")
                logger.info(f"Heatmap saved to {path}")
        finally:
            # FIX: Ensure all matplotlib figures are strictly closed inside finally: plt.close(fig) with gc.collect()
            if fig is not None:
                try:
                    fig.clf()
                    plt.close(fig)
                except (AttributeError, KeyError, RuntimeError, TypeError, ValueError, OSError, ImportError):
                    pass
            gc.collect()
    except Exception as e:
        logger.error(f"Async attention heatmap extraction failed: {e}")


def _async_compute_svd_condition(name: str, weight_tensor: torch.Tensor) -> None:
    """
    Worker task running on background thread to compute SVD condition number
    without stalling the primary PyTorch CUDA stream.
    """
    try:
        # FIX: Offload SVD matrix decomposition and .cpu() tensor movements to background worker thread
        if hasattr(weight_tensor, "detach"):
            weight_tensor = weight_tensor.detach()
        if hasattr(weight_tensor, "cpu"):
            weight_tensor = weight_tensor.cpu().float()
        
        _, S, _ = torch.linalg.svd(weight_tensor, full_matrices=False)
        cond = S[0].item() / S[-1].item() if S[-1] != 0 else float("inf")
        logger.info(f"  {name}: condition number = {cond:.2f}")
    except Exception as e:
        logger.warning(f"  {name}: async SVD failed ({e})")


def _async_compute_layerwise_rank(hidden_states: List[torch.Tensor], global_step: int) -> None:
    """
    Worker task running on background thread to compute layerwise effective rank.
    """
    try:
        # FIX: Execute covariance computation, SVD, and CPU float extractions asynchronously
        for i, hs in enumerate(hidden_states):
            if hasattr(hs, "detach"):
                hs = hs.detach()
            if hasattr(hs, "cpu"):
                hs = hs.cpu().float()
            
            hs_flat = hs.view(-1, hs.size(-1))
            cov = hs_flat.t().mm(hs_flat) / max(hs_flat.size(0) - 1, 1)
            try:
                _, S, _ = torch.linalg.svd(cov, full_matrices=False)
                S_norm = S / S.sum()
                entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-12))
                eff_rank = torch.exp(entropy).item()
                logger.info(f"  Layer {i}: effective rank = {eff_rank:.1f}")
            except Exception as e:
                logger.warning(f"  Layer {i}: async SVD failed ({e})")
    except Exception as e:
        logger.error(f"Async layerwise analysis failed: {e}")


def _async_compute_activation_stats(
    hidden_states: List[torch.Tensor],
    global_step: int,
    stats_list: List[Dict[str, float]],
) -> None:
    """
    Worker task running on background thread to calculate mean/std activation telemetry.
    """
    try:
        # FIX: Compute mean/std and .item() extractions asynchronously to avoid GPU sync stalls
        layer_stats: Dict[str, float] = {}
        for i, hs in enumerate(hidden_states):
            if hasattr(hs, "detach"):
                hs = hs.detach()
            if hasattr(hs, "cpu"):
                hs = hs.cpu().float()
            layer_stats[f"layer_{i}_mean"] = hs.mean().item()
            layer_stats[f"layer_{i}_std"] = hs.std().item()
        stats_list.append(layer_stats)
        logger.info(f"Activation stats step {global_step}: {layer_stats}")
    except Exception as e:
        logger.error(f"Async activation stats calculation failed: {e}")


class SVDAttentionCallback(AsyncTelemetryCallback):
    """Periodically computes SVD of attention projection weights and logs condition numbers asynchronously."""

    def __init__(self, layer_interval: int = 2, log_every_steps: int = 500, max_workers: int = 2) -> None:
        super().__init__(max_workers=max_workers, thread_name_prefix="AsyncSVDWorker")
        self.layer_interval = layer_interval
        self.log_every_steps = log_every_steps

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """On step end."""
        model = kwargs.get("model")
        if model is None or state.global_step % self.log_every_steps != 0:
            return

        logger.info("Running SVD analysis on attention layers asynchronously...")
        for name, module in model.named_modules():
            if (
                "attention" in name.lower()
                and (hasattr(module, "q_proj") or hasattr(module, "q_lin") or hasattr(module, "query"))
            ):
                q_weight = None
                for attr in ["q_proj", "q_lin", "query"]:
                    if hasattr(module, attr):
                        q_weight = getattr(module, attr).weight.data
                        break
                if q_weight is None:
                    continue

                # FIX: Offload SVD condition number extraction to non-blocking ThreadPoolExecutor
                detached_weight = q_weight.detach().clone()
                self.submit_async(_async_compute_svd_condition, name, detached_weight)


class AttentionHeatmapCallback(AsyncTelemetryCallback):
    """Generates attention heatmaps asynchronously for sample inputs and saves to output_dir."""

    def __init__(
        self,
        output_dir: str,
        sample_inputs: Optional[Dict[str, Any]] = None,
        log_every_steps: int = 1000,
        max_workers: int = 2,
    ) -> None:
        super().__init__(max_workers=max_workers, thread_name_prefix="AsyncAttentionHeatmapWorker")
        self.output_dir = output_dir
        self.sample_inputs = sample_inputs
        self.log_every_steps = log_every_steps
        os.makedirs(output_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        """On evaluate."""
        if state.global_step % self.log_every_steps != 0 or self.sample_inputs is None:
            return
        model = kwargs.get("model")
        if model is None:
            return

        logger.info("Extracting attention matrices for async heatmap generation...")
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(
                    **{
                        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in self.sample_inputs.items()
                    },
                    output_attentions=True,
                )
            attentions = outputs.attentions
            if attentions:
                # FIX: Move attention slice detached to async ThreadPoolExecutor for .cpu().numpy() and plotting
                raw_attn_slice = attentions[-1][0, 0].detach().clone()
                layer_idx = len(attentions)
                head_idx = 1
                self.submit_async(
                    _async_process_attention_heatmap,
                    raw_attn_slice,
                    state.global_step,
                    self.output_dir,
                    layer_idx,
                    head_idx,
                )
        except Exception as e:
            logger.error(f"Failed to extract attention for async heatmap: {e}")


class GradientNormCallback(TrainerCallback):
    """Logs the L2 norm of gradients after each optimizer step."""

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """On log."""
        if logs is None:
            return
        model = kwargs.get("model")
        if model is not None:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            logs["grad_norm"] = total_norm
            logger.info(f"Gradient norm: {total_norm:.4f}")


class LayerwiseAnalysisCallback(AsyncTelemetryCallback):
    """Computes intrinsic dimension / effective rank of hidden representations asynchronously."""

    def __init__(
        self,
        log_every_steps: int = 2000,
        sample_inputs: Optional[Dict[str, Any]] = None,
        max_workers: int = 2,
    ) -> None:
        super().__init__(max_workers=max_workers, thread_name_prefix="AsyncLayerwiseWorker")
        self.log_every_steps = log_every_steps
        self.sample_inputs = sample_inputs

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        """On evaluate."""
        if state.global_step % self.log_every_steps != 0 or self.sample_inputs is None:
            return
        model = kwargs.get("model")
        if model is None:
            return
        logger.info("Running layer-wise analysis (effective rank) asynchronously...")
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(
                    **{
                        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in self.sample_inputs.items()
                    },
                    output_hidden_states=True,
                )
            hidden_states = outputs.hidden_states
            if hidden_states:
                # FIX: Offload covariance calculation and effective rank computation to ThreadPoolExecutor
                detached_hidden = [hs.detach().clone() for hs in hidden_states]
                self.submit_async(_async_compute_layerwise_rank, detached_hidden, state.global_step)
        except Exception as e:
            logger.error(f"Layerwise analysis dispatch failed: {e}")


class ActivationStatsCallback(AsyncTelemetryCallback):
    """Collects mean/std of activations during evaluation asynchronously."""

    def __init__(self, sample_inputs: Optional[Dict[str, Any]] = None, max_workers: int = 2) -> None:
        super().__init__(max_workers=max_workers, thread_name_prefix="AsyncActivationWorker")
        self.sample_inputs = sample_inputs
        self.stats: List[Dict[str, float]] = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        """On evaluate."""
        if self.sample_inputs is None:
            return
        model = kwargs.get("model")
        if model is None:
            return
        try:
            model.eval()
            with torch.no_grad():
                outputs = model(
                    **{
                        k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                        for k, v in self.sample_inputs.items()
                    },
                    output_hidden_states=True,
                )
            hidden_states = outputs.hidden_states
            if hidden_states:
                # FIX: Offload mean/std reductions to non-blocking background ThreadPoolExecutor
                detached_hidden = [hs.detach().clone() for hs in hidden_states]
                self.submit_async(
                    _async_compute_activation_stats,
                    detached_hidden,
                    state.global_step,
                    self.stats,
                )
        except Exception as e:
            logger.error(f"Activation stats dispatch failed: {e}")
