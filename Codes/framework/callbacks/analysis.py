"""Analysis callbacks: SVD, attention heatmaps, gradient norms, activation stats."""

import logging
import os

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class SVDAttentionCallback(TrainerCallback):
    """Periodically computes SVD of attention projection weights and logs condition numbers."""

    def __init__(self, layer_interval: int = 2, log_every_steps: int = 500):
        self.layer_interval = layer_interval
        self.log_every_steps = log_every_steps

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or state.global_step % self.log_every_steps != 0:
            return

        logger.info("Running SVD analysis on attention layers...")
        for name, module in model.named_modules():
            if (
                "attention" in name.lower()
                and hasattr(module, "q_proj")
                or hasattr(module, "q_lin")
            ):
                # Extract Q weight (try q_proj first, then q_lin)
                q_weight = None
                for attr in ["q_proj", "q_lin", "query"]:
                    if hasattr(module, attr):
                        q_weight = getattr(module, attr).weight.data
                        break
                if q_weight is None:
                    continue
                # Compute SVD
                try:
                    _, S, _ = torch.linalg.svd(q_weight.float(), full_matrices=False)
                    cond = S[0].item() / S[-1].item() if S[-1] != 0 else float("inf")
                    logger.info(f"  {name}: condition number = {cond:.2f}")
                except Exception as e:
                    logger.warning(f"  {name}: SVD failed ({e})")


class AttentionHeatmapCallback(TrainerCallback):
    """Generates attention heatmaps for a sample input and saves to output_dir."""

    def __init__(
        self, output_dir: str, sample_inputs: dict = None, log_every_steps: int = 1000
    ):
        self.output_dir = output_dir
        self.sample_inputs = sample_inputs  # dict with input_ids, attention_mask, token_type_ids (optional)
        self.log_every_steps = log_every_steps
        os.makedirs(output_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.global_step % self.log_every_steps != 0 or self.sample_inputs is None:
            return
        model = kwargs.get("model")
        if model is None:
            return
        logger.info("Generating attention heatmaps...")
        try:
            import matplotlib.pyplot as plt

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
            # Plot the last layer, first head
            att = attentions[-1][0, 0].cpu().numpy()
            plt.figure(figsize=(10, 8))
            plt.imshow(att, cmap="viridis")
            plt.title(
                f"Attention Heatmap – Layer {len(attentions)} Head 1 – Step {state.global_step}"
            )
            plt.colorbar()
            path = os.path.join(
                self.output_dir, f"attn_heatmap_step{state.global_step}.png"
            )
            plt.savefig(path, dpi=150)
            plt.close()
            logger.info(f"Heatmap saved to {path}")
        except Exception as e:
            logger.error(f"Failed to generate heatmap: {e}")


class GradientNormCallback(TrainerCallback):
    """Logs the L2 norm of gradients after each optimizer step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
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


class LayerwiseAnalysisCallback(TrainerCallback):
    """Computes intrinsic dimension / effective rank of hidden representations."""

    def __init__(self, log_every_steps: int = 2000, sample_inputs: dict = None):
        self.log_every_steps = log_every_steps
        self.sample_inputs = sample_inputs

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if state.global_step % self.log_every_steps != 0 or self.sample_inputs is None:
            return
        model = kwargs.get("model")
        if model is None:
            return
        logger.info("Running layer‑wise analysis (effective rank)...")
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
            for i, hs in enumerate(hidden_states):
                # Flatten to (batch*seq, hidden)
                hs_flat = hs.view(-1, hs.size(-1))
                # Compute singular values via covariance matrix
                cov = hs_flat.t().mm(hs_flat) / (hs_flat.size(0) - 1)
                try:
                    _, S, _ = torch.linalg.svd(cov, full_matrices=False)
                    # Effective rank = exp(entropy)
                    S_norm = S / S.sum()
                    entropy = -torch.sum(S_norm * torch.log(S_norm + 1e-12))
                    eff_rank = torch.exp(entropy).item()
                    logger.info(f"  Layer {i}: effective rank = {eff_rank:.1f}")
                except Exception as e:
                    logger.warning(f"  Layer {i}: SVD failed ({e})")
        except Exception as e:
            logger.error(f"Layerwise analysis failed: {e}")


class ActivationStatsCallback(TrainerCallback):
    """Collects mean/std of activations during evaluation."""

    def __init__(self, sample_inputs: dict = None):
        self.sample_inputs = sample_inputs
        self.stats = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.sample_inputs is None:
            return
        model = kwargs.get("model")
        if model is None:
            return
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
        layer_stats = {}
        for i, hs in enumerate(hidden_states):
            layer_stats[f"layer_{i}_mean"] = hs.mean().item()
            layer_stats[f"layer_{i}_std"] = hs.std().item()
        self.stats.append(layer_stats)
        logger.info(f"Activation stats step {state.global_step}: {layer_stats}")
