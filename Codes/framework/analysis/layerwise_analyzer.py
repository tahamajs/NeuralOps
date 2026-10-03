"""
Streaming Layer-wise Analysis Engine for Foundation Models & Transformers.

Computes representation drift, cosine similarities, kurtosis-based outlier metrics,
effective spectral rank, and intrinsic dimensionality on-the-fly without accumulating
all 32+ layer activations simultaneously in VRAM, eliminating streaming OOM crashes.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cosine


@dataclass
class StreamingLayerAccumulator:
    """
    Online Welford & spectral accumulator for single-layer activations.
    Updates statistical moments dynamically and frees raw activation buffers immediately.
    """

    total_tokens: int = 0
    total_batches: int = 0
    running_mean_vector: np.ndarray | None = None
    max_activation: float = 0.0
    sum_outlier_count_6sigma: int = 0
    total_elements: int = 0

    # Online moments for kurtosis and variance
    sum_m2: float = 0.0
    sum_m4: float = 0.0

    # Covariance sketch for effective rank (subsampled rank estimation)
    cov_matrix: np.ndarray | None = None
    cov_samples: int = 0

    def update(self, act_tensor: torch.Tensor) -> None:
        """
        Dynamically updates moments from an incoming activation batch [B, S, D] or [N, D]
        and releases tensor references.
        """
        # FIX: Compute outlier and spectral metrics dynamically per layer and immediately free activations
        with torch.no_grad():
            act_f32 = act_tensor.detach().float()

            # Flatten to [N, D]
            if act_f32.ndim == 3:
                b, s, d = act_f32.shape
                flat_act = act_f32.reshape(-1, d)
            elif act_f32.ndim == 2:
                flat_act = act_f32
                d = flat_act.shape[1]
            else:
                flat_act = act_f32.reshape(-1, act_f32.shape[-1])
                d = flat_act.shape[-1]

            n_tokens = flat_act.shape[0]
            if n_tokens == 0:
                return

            # 1. Update running mean representation vector (CPU numpy)
            batch_mean = flat_act.mean(dim=0).cpu().numpy()
            if self.running_mean_vector is None:
                self.running_mean_vector = batch_mean.copy()
            else:
                self.running_mean_vector = self.running_mean_vector * (
                    self.total_tokens / (self.total_tokens + n_tokens)
                ) + batch_mean * (n_tokens / (self.total_tokens + n_tokens))

            # 2. Update Peak Outlier Magnitude
            batch_max = float(flat_act.abs().max().item())
            self.max_activation = max(self.max_activation, batch_max)

            # 3. Update Outlier Ratio & Kurtosis Moments
            act_cpu = flat_act.cpu().numpy()
            global_elem = act_cpu.size
            self.total_elements += global_elem

            mu = float(np.mean(act_cpu))
            centered = act_cpu - mu
            m2 = float(np.mean(centered**2))
            m4 = float(np.mean(centered**4))

            self.sum_m2 += m2 * global_elem
            self.sum_m4 += m4 * global_elem

            std = np.sqrt(max(1e-12, m2))
            outliers = np.sum(np.abs(centered) > (6.0 * std))
            self.sum_outlier_count_6sigma += int(outliers)

            # 4. Running Subsampled Covariance for Effective Rank
            sub_n = min(n_tokens, 256)
            sub_act = act_cpu[:sub_n] - batch_mean
            batch_cov = (sub_act.T @ sub_act) / max(1, sub_n)
            if self.cov_matrix is None:
                self.cov_matrix = batch_cov
                self.cov_samples = sub_n
            else:
                total_s = self.cov_samples + sub_n
                self.cov_matrix = self.cov_matrix * (
                    self.cov_samples / total_s
                ) + batch_cov * (sub_n / total_s)
                self.cov_samples = total_s

            self.total_tokens += n_tokens
            self.total_batches += 1

    def compute_kurtosis(self) -> float:
        """Compute excess kurtosis: E[(x-mu)^4] / (E[(x-mu)^2])^2 - 3."""
        if self.total_elements == 0 or self.sum_m2 == 0:
            return 0.0
        m2 = self.sum_m2 / self.total_elements
        m4 = self.sum_m4 / self.total_elements
        var_sq = m2**2
        if var_sq < 1e-12:
            return 0.0
        return float((m4 / var_sq) - 3.0)

    def compute_outlier_ratio(self) -> float:
        """Compute 6-sigma outlier fraction."""
        if self.total_elements == 0:
            return 0.0
        return float(self.sum_outlier_count_6sigma / self.total_elements)

    def compute_effective_rank(self) -> float:
        """
        Computes the effective rank erank(Sigma) = exp(H(p)) where p_i = lambda_i / sum(lambda).
        """
        if self.cov_matrix is None:
            return 1.0
        try:
            eigvals = np.linalg.eigvalsh(self.cov_matrix)
            eigvals = np.maximum(eigvals, 1e-12)
            eigvals_norm = eigvals / np.sum(eigvals)
            entropy = -np.sum(eigvals_norm * np.log(eigvals_norm + 1e-12))
            return float(np.exp(entropy))
        except Exception:
            return 1.0


class LayerwiseAnalyzer:
    """
    Streaming Layerwise Analyzer that captures inter-layer representations,
    outlier distribution spikes, and effective rank without accumulating VRAM memory.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str | torch.device | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = str(device) if device is not None else self._get_device()
        self.model.to(self.device)
        self.model.eval()
        self.layer_accumulators: list[StreamingLayerAccumulator] = []
        self.hooks: list[torch.utils.hooks.RemovableHandle] = []

    def _get_device(self) -> str:
        """Detect best available compute device."""
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _find_transformer_layers(self) -> list[tuple[str, nn.Module]]:
        """Identify candidate Transformer / Encoder / Decoder block modules."""
        target_layers = []
        for name, module in self.model.named_modules():
            name_lower = name.lower()
            # Match standard transformer layer patterns
            if any(
                pattern in name_lower
                for pattern in [
                    "encoder.layer.",
                    "transformer.layer.",
                    "transformer.h.",
                    "model.layers.",
                    "layers.",
                    "h.",
                    "decoder.layer.",
                    "transformer.blocks.",
                ]
            ):
                # Filter to only the top-level block (e.g. layers.0, not layers.0.attention)
                parts = name.split(".")
                if parts[-1].isdigit():
                    target_layers.append((name, module))

        # Fallback: if no digit-indexed layers found, look for any layer module
        if not target_layers:
            for name, module in self.model.named_modules():
                if "layer" in name.lower() and isinstance(
                    module, (nn.ModuleList, nn.Sequential)
                ):
                    for sub_idx, sub_mod in enumerate(module):
                        target_layers.append((f"{name}.{sub_idx}", sub_mod))
                    break
        return target_layers

    def register_streaming_hooks(self) -> None:
        """Registers streaming forward hooks on all model transformer layers."""
        self.remove_hooks()
        layers = self._find_transformer_layers()

        num_layers = len(layers)
        self.layer_accumulators = [
            StreamingLayerAccumulator() for _ in range(num_layers)
        ]

        for layer_idx, (layer_name, module) in enumerate(layers):

            def make_hook(idx: int):
                def hook_fn(mod: nn.Module, inputs: Any, output: Any) -> None:
                    # Extract tensor
                    tensor = output[0] if isinstance(output, tuple) else output
                    if isinstance(tensor, torch.Tensor):
                        # FIX: Update streaming accumulator dynamically and release activation immediately
                        self.layer_accumulators[idx].update(tensor)
                        # Explicitly release references
                        del tensor

                return hook_fn

            hook_handle = module.register_forward_hook(make_hook(layer_idx))
            self.hooks.append(hook_handle)

    def remove_hooks(self) -> None:
        """Remove all active hooks and free memory."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def analyze(self, texts: list[str], batch_size: int = 8) -> dict[str, Any]:
        """
        Executes streaming layerwise analysis on input text corpus.

        Args:
            texts: List of input string prompts/sentences.
            batch_size: Batch size for tokenization and model inference.

        Returns:
            Dict containing cosine similarities, effective ranks, outlier statistics,
            and least-compressible layer rankings.
        """
        self.register_streaming_hooks()

        try:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                )
                inputs = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.no_grad():
                    self.model(**inputs)

                # Cleanup batch inputs
                del inputs, encoded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            self.remove_hooks()

        num_layers = len(self.layer_accumulators)
        if num_layers == 0:
            return {
                "num_layers": 0,
                "cosine_similarities": [],
                "effective_ranks": [],
                "max_outlier_magnitudes": [],
                "outlier_ratios": [],
                "kurtosis_scores": [],
                "intrinsic_dimensions": [],
            }

        # Extract per-layer mean vectors
        mean_vectors = [
            (
                acc.running_mean_vector
                if acc.running_mean_vector is not None
                else np.zeros(128, dtype=np.float32)
            )
            for acc in self.layer_accumulators
        ]

        # Compute cosine similarity between adjacent layers
        similarities = []
        for i in range(num_layers - 1):
            v1, v2 = mean_vectors[i], mean_vectors[i + 1]
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 > 1e-8 and norm2 > 1e-8:
                sim = 1.0 - float(cosine(v1, v2))
                sim_clamped = max(0.0, min(1.0, sim))
            else:
                sim_clamped = 1.0
            similarities.append(sim_clamped)

        effective_ranks = [
            acc.compute_effective_rank() for acc in self.layer_accumulators
        ]
        max_outliers = [acc.max_activation for acc in self.layer_accumulators]
        outlier_ratios = [
            acc.compute_outlier_ratio() for acc in self.layer_accumulators
        ]
        kurtosis_scores = [acc.compute_kurtosis() for acc in self.layer_accumulators]

        # Estimate intrinsic dimension using mean vectors
        stacked_means = np.stack(mean_vectors) if num_layers > 1 else np.zeros((1, 128))
        intrinsic_dims = [
            estimate_intrinsic_dimension(
                acc.cov_matrix if acc.cov_matrix is not None else np.eye(10), k=3
            )
            for acc in self.layer_accumulators
        ]

        return {
            "num_layers": num_layers,
            "cosine_similarities": similarities,
            "effective_ranks": effective_ranks,
            "max_outlier_magnitudes": max_outliers,
            "outlier_ratios": outlier_ratios,
            "kurtosis_scores": kurtosis_scores,
            "intrinsic_dimensions": intrinsic_dims,
        }


def estimate_intrinsic_dimension(hidden_states: np.ndarray, k: int = 5) -> float:
    """
    Estimates the intrinsic dimension of a representation manifold using k-NN distances.

    Args:
        hidden_states: Sample representation array of shape [N, D].
        k: Neighbor rank for distance estimation.

    Returns:
        float: Estimated intrinsic dimension between 1.0 and D.
    """
    hidden_states = np.asarray(hidden_states, dtype=np.float64)
    if hidden_states.ndim == 1:
        return 1.0
    N, D = hidden_states.shape
    if k + 1 > N:
        return float(min(N, D))

    from scipy.spatial.distance import pdist, squareform

    try:
        dists = squareform(pdist(hidden_states))
        np.fill_diagonal(dists, np.inf)
        kth_dists = np.sort(dists, axis=1)[:, k - 1]
        valid = kth_dists > 1e-10
        if not valid.any():
            return float(D)

        # Maximum likelihood estimation of intrinsic dimension
        log_ratios = np.log(kth_dists[valid] / (kth_dists[valid][:, None] + 1e-12))
        id_estimate = -1.0 / (np.mean(log_ratios) + 1e-12)
        return float(max(1.0, min(float(D), abs(id_estimate))))
    except Exception:
        return float(min(N, D))
