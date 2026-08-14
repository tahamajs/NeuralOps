"""
Computation, Weight & Memory Studios for Transformer Architecture Analysis.

Provides:
- FFNNeuronExplorer: Neuron activation heatmaps, top-activating examples, dead neuron detection, SAE feature breakdown.
- KVCacheMonitor: Token-by-token memory scaling, Delta-KV temporal heatmaps, attention sink detection, compression tracking.
- WeightMatrixDecomposer: SVD, CUR, and RPCA low-rank matrix factorizations with energy retention profiling.
- LoRAAdapterStudio: Rank spectrum analyzer, delta-W magnitude inspector, low-rank gradient dynamics, SVD vs Zero init comparison.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from .base import VISUALIZER_REGISTRY, BaseVisualizer
from .utils import create_custom_colormap, set_plot_style


def _to_numpy(data: Union[np.ndarray, torch.Tensor, List[Any]]) -> np.ndarray:
    """Helper to convert PyTorch tensors or lists into float32 numpy arrays."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy().astype(np.float32)
    if isinstance(data, list):
        return np.array(data, dtype=np.float32)
    if isinstance(data, np.ndarray):
        return data.astype(np.float32)
    return np.asarray(data, dtype=np.float32)


# =============================================================================
# 1. FFN / MLP NEURON EXPLORER
# =============================================================================

@VISUALIZER_REGISTRY.register("ffn_neuron_explorer")
class FFNNeuronExplorer(BaseVisualizer):
    """
    FFN/MLP Explorer — Neuron Activation Universe.
    Analyzes intermediate activations of feed-forward sublayers (mlp.post_act).
    """

    def __init__(self, output_dir: str = "visualizations/ffn_explorer") -> None:
        super().__init__(output_dir)
        self.cmap = create_custom_colormap()

    def generate(self, activations: Union[np.ndarray, torch.Tensor], tokens: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Default generation pipeline for FFN exploration."""
        act_np = _to_numpy(activations)
        dead_info = self.detect_dead_neurons(act_np)
        stats = self.compute_neuron_statistics(act_np)
        heatmap_res = self.generate_activation_heatmap(act_np, tokens=tokens, save=kwargs.get("save", True))
        return {
            "dead_neurons": dead_info,
            "statistics": stats,
            "heatmap": heatmap_res,
        }

    def generate_activation_heatmap(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        tokens: Optional[List[str]] = None,
        layer_idx: int = 0,
        neuron_indices: Optional[List[int]] = None,
        max_neurons: int = 64,
        title: Optional[str] = None,
        save: bool = True,
        show: bool = False,
    ) -> Dict[str, Any]:
        """
        Generates token x neuron activation heatmap matrix.
        Supports 2D [seq_len, d_mlp] or 3D [batch_size, seq_len, d_mlp].
        """
        act = _to_numpy(activations)
        if act.ndim == 3:
            act = np.mean(act, axis=0)  # average across batch

        seq_len, d_mlp = act.shape
        if tokens is None:
            tokens = [f"tok_{i}" for i in range(seq_len)]
        elif len(tokens) < seq_len:
            tokens = tokens + [f"tok_{i}" for i in range(len(tokens), seq_len)]
        elif len(tokens) > seq_len:
            tokens = tokens[:seq_len]

        if neuron_indices is not None:
            selected_indices = [idx for idx in neuron_indices if 0 <= idx < d_mlp]
        else:
            if d_mlp > max_neurons:
                # Select top variance neurons
                var_per_neuron = np.var(act, axis=0)
                selected_indices = np.argsort(var_per_neuron)[::-1][:max_neurons].tolist()
                selected_indices.sort()
            else:
                selected_indices = list(range(d_mlp))

        sub_matrix = act[:, selected_indices]

        fig, ax = plt.subplots(figsize=(max(8, len(selected_indices) * 0.25), max(6, seq_len * 0.3)))
        sns.heatmap(
            sub_matrix,
            xticklabels=[f"N_{i}" for i in selected_indices],
            yticklabels=tokens,
            cmap="viridis",
            cbar_kws={"label": "Activation Magnitude"},
            ax=ax,
        )
        plot_title = title or f"FFN Neuron Activations (Layer {layer_idx}) - {len(selected_indices)} Neurons"
        ax.set_title(plot_title)
        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Token Sequence")
        plt.tight_layout()

        filename = f"ffn_heatmap_layer{layer_idx}_{seq_len}x{len(selected_indices)}.png"
        saved_path = self._save_figure(fig, filename) if save else None
        if show:
            plt.show()

        return {
            "matrix": sub_matrix.tolist(),
            "selected_neurons": selected_indices,
            "tokens": tokens,
            "shape": list(sub_matrix.shape),
            "file_path": saved_path,
            "mean_activation": float(np.mean(sub_matrix)),
            "max_activation": float(np.max(sub_matrix)),
            "sparsity_ratio": float(np.mean(sub_matrix <= 1e-6)),
        }

    def detect_dead_neurons(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        threshold: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        Detects inactive/dead neurons where max activation across all tokens < threshold.
        """
        act = _to_numpy(activations)
        if act.ndim == 3:
            act = act.reshape(-1, act.shape[-1])
        elif act.ndim == 1:
            act = act.reshape(1, -1)

        total_neurons = act.shape[-1]
        max_per_neuron = np.max(np.abs(act), axis=0)
        mean_per_neuron = np.mean(act, axis=0)
        var_per_neuron = np.var(act, axis=0)

        dead_mask = max_per_neuron < threshold
        dead_indices = np.where(dead_mask)[0].tolist()
        dead_count = len(dead_indices)
        dead_ratio = float(dead_count / total_neurons) if total_neurons > 0 else 0.0

        if dead_ratio < 0.05:
            health = "healthy"
        elif dead_ratio < 0.20:
            health = "warning"
        else:
            health = "critical"

        return {
            "dead_indices": dead_indices,
            "dead_count": dead_count,
            "alive_count": total_neurons - dead_count,
            "total_neurons": total_neurons,
            "dead_ratio": dead_ratio,
            "health": health,
            "max_activations": max_per_neuron.tolist(),
            "mean_activations": mean_per_neuron.tolist(),
            "variances": var_per_neuron.tolist(),
        }

    def find_top_activating_examples(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        tokens_or_texts: List[str],
        top_k: int = 5,
        neuron_indices: Optional[List[int]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Discovers top-k tokens/snippets that maximally activate each neuron.
        Calculates selectivity score = 1 - (entropy / log(N)).
        """
        act = _to_numpy(activations)
        if act.ndim == 3:
            act = np.mean(act, axis=0)

        seq_len, d_mlp = act.shape
        indices_to_process = neuron_indices if neuron_indices is not None else list(range(min(d_mlp, 32)))

        results = {}
        for n_idx in indices_to_process:
            if not (0 <= n_idx < d_mlp):
                continue
            col = act[:, n_idx]
            top_k_indices = np.argsort(col)[::-1][:top_k]
            top_examples = []
            for rank_pos, idx in enumerate(top_k_indices):
                tok_str = tokens_or_texts[idx] if idx < len(tokens_or_texts) else f"pos_{idx}"
                top_examples.append({
                    "rank": rank_pos + 1,
                    "token_index": int(idx),
                    "token": tok_str,
                    "activation": float(col[idx]),
                })

            # Selectivity calculation
            pos_act = np.maximum(col, 0.0)
            sum_pos = np.sum(pos_act)
            if sum_pos > 1e-9:
                p = pos_act / sum_pos
                p_nonzero = p[p > 1e-12]
                entropy = -np.sum(p_nonzero * np.log(p_nonzero))
                max_entropy = np.log(len(col)) if len(col) > 1 else 1.0
                selectivity = float(max(0.0, 1.0 - (entropy / max_entropy)))
            else:
                selectivity = 0.0

            results[n_idx] = {
                "neuron_index": n_idx,
                "top_examples": top_examples,
                "max_activation": float(np.max(col)),
                "mean_activation": float(np.mean(col)),
                "variance": float(np.var(col)),
                "selectivity": selectivity,
            }

        return results

    def sae_feature_breakdown(
        self,
        activations: Union[np.ndarray, torch.Tensor],
        sae_encoder_weight: Union[np.ndarray, torch.Tensor],
        sae_bias: Optional[Union[np.ndarray, torch.Tensor]] = None,
        sae_decoder_weight: Optional[Union[np.ndarray, torch.Tensor]] = None,
        top_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Sparse Autoencoder (SAE) monosemantic feature breakdown:
        f = ReLU(x W_enc + b_enc)
        Computes L0 sparsity, active feature counts, and reconstruction error.
        """
        X = _to_numpy(activations)
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
        elif X.ndim == 1:
            X = X.reshape(1, -1)

        W_enc = _to_numpy(sae_encoder_weight)
        b_enc = _to_numpy(sae_bias) if sae_bias is not None else np.zeros(W_enc.shape[-1], dtype=np.float32)

        # Feature activations: ReLU(X @ W_enc + b_enc)
        features = np.maximum(0.0, X @ W_enc + b_enc)  # [tokens, n_features]
        n_tokens, n_features = features.shape

        # L0 sparsity: average active features per token
        active_counts_per_token = np.sum(features > 1e-6, axis=1)
        mean_l0 = float(np.mean(active_counts_per_token))
        sparsity_ratio = float(1.0 - (mean_l0 / n_features)) if n_features > 0 else 1.0

        # Global feature importance
        mean_feature_act = np.mean(features, axis=0)
        top_feature_indices = np.argsort(mean_feature_act)[::-1][:top_features].tolist()

        feature_summary = []
        for idx in top_feature_indices:
            feature_summary.append({
                "feature_index": int(idx),
                "mean_activation": float(mean_feature_act[idx]),
                "firing_rate": float(np.mean(features[:, idx] > 1e-6)),
                "max_activation": float(np.max(features[:, idx])),
            })

        # Reconstruction error if decoder is provided
        recon_error = None
        if sae_decoder_weight is not None:
            W_dec = _to_numpy(sae_decoder_weight)
            X_recon = features @ W_dec
            frobenius_err = np.linalg.norm(X - X_recon)
            orig_norm = np.linalg.norm(X)
            recon_error = float(frobenius_err / (orig_norm + 1e-9))

        return {
            "num_features": n_features,
            "mean_l0_sparsity": mean_l0,
            "sparsity_ratio": sparsity_ratio,
            "top_features": feature_summary,
            "reconstruction_error": recon_error,
            "feature_matrix_shape": list(features.shape),
        }

    def compute_neuron_statistics(self, activations: Union[np.ndarray, torch.Tensor]) -> Dict[str, Any]:
        """
        Computes statistical moments: mean, variance, skewness, kurtosis for neurons.
        """
        act = _to_numpy(activations)
        if act.ndim == 3:
            act = act.reshape(-1, act.shape[-1])
        elif act.ndim == 1:
            act = act.reshape(1, -1)

        mean_vals = np.mean(act, axis=0)
        var_vals = np.var(act, axis=0)
        std_vals = np.std(act, axis=0) + 1e-9

        # Higher moments
        diff = act - mean_vals
        skewness = np.mean((diff / std_vals) ** 3, axis=0)
        kurtosis = np.mean((diff / std_vals) ** 4, axis=0) - 3.0  # excess kurtosis

        return {
            "mean_overall": float(np.mean(mean_vals)),
            "variance_overall": float(np.mean(var_vals)),
            "mean_per_neuron": mean_vals.tolist(),
            "variance_per_neuron": var_vals.tolist(),
            "skewness_per_neuron": skewness.tolist(),
            "kurtosis_per_neuron": kurtosis.tolist(),
            "max_val": float(np.max(act)),
            "min_val": float(np.min(act)),
        }

    def plot_dead_neuron_distribution(
        self,
        layer_activations_dict: Dict[int, Union[np.ndarray, torch.Tensor]],
        threshold: float = 1e-6,
        save: bool = True,
    ) -> Optional[str]:
        """
        Plots dead neuron ratio and counts across layers.
        """
        layers = sorted(layer_activations_dict.keys())
        dead_ratios = []
        dead_counts = []
        for l in layers:
            info = self.detect_dead_neurons(layer_activations_dict[l], threshold=threshold)
            dead_ratios.append(info["dead_ratio"] * 100.0)
            dead_counts.append(info["dead_count"])

        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        bars = ax1.bar(layers, dead_ratios, alpha=0.7, color="#E74C3C", label="Dead Ratio (%)")
        ax2.plot(layers, dead_counts, "o--", color="#2C3E50", linewidth=2, label="Dead Count")

        ax1.set_xlabel("Layer Index")
        ax1.set_ylabel("Dead Neuron Ratio (%)", color="#E74C3C")
        ax2.set_ylabel("Dead Neuron Count", color="#2C3E50")
        ax1.set_title("Dead Neuron Distribution Across Transformer Layers")
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = "dead_neuron_layer_distribution.png"
        return self._save_figure(fig, filename) if save else None


# =============================================================================
# 2. KV CACHE MONITOR
# =============================================================================

@VISUALIZER_REGISTRY.register("kv_cache_monitor")
class KVCacheMonitor(BaseVisualizer):
    """
    KV Cache Monitor — Token-by-Token Memory Scaling & Dynamic Retention.
    Tracks memory growth, Delta-KV temporal differences, and attention sinks.
    """

    def __init__(self, output_dir: str = "visualizations/kv_monitor") -> None:
        super().__init__(output_dir)

    def generate(self, seq_lens: Optional[List[int]] = None, **kwargs) -> Dict[str, Any]:
        """Default generation pipeline for KV cache monitoring."""
        if seq_lens is None:
            seq_lens = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        scaling_info = self.simulate_cache_growth(
            seq_lens=seq_lens,
            num_layers=kwargs.get("num_layers", 32),
            num_heads=kwargs.get("num_heads", 32),
            head_dim=kwargs.get("head_dim", 128),
            bytes_per_elem=kwargs.get("bytes_per_elem", 2.0),
        )
        plot_path = self.plot_memory_scaling(seq_lens, scaling_info["compression_curves_mb"], save=kwargs.get("save", True))
        return {
            "scaling": scaling_info,
            "plot_path": plot_path,
        }

    def simulate_cache_growth(
        self,
        seq_lens: List[int],
        num_layers: int = 32,
        num_heads: int = 32,
        head_dim: int = 128,
        bytes_per_elem: float = 2.0,
        compression_methods: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Computes token-by-token KV cache memory scaling:
        Memory (bytes) = 2 * L * H * N * d_head * bytes_per_elem
        """
        if compression_methods is None:
            compression_methods = {
                "DeltaKV (r=16)": 0.29,
                "H2O (Heavy-Hitter)": 0.35,
                "StreamingLLM (Sink+Local)": 0.22,
                "SnapKV": 0.30,
            }

        baseline_mb = []
        baseline_gb = []
        curves_mb: Dict[str, List[float]] = {"Full Baseline (Dense)": []}
        for name in compression_methods:
            curves_mb[name] = []

        for n in seq_lens:
            # 2 for Key and Value
            bytes_total = 2 * num_layers * num_heads * n * head_dim * bytes_per_elem
            mb = bytes_total / (1024 * 1024)
            gb = bytes_total / (1024 * 1024 * 1024)
            baseline_mb.append(round(mb, 2))
            baseline_gb.append(round(gb, 3))
            curves_mb["Full Baseline (Dense)"].append(round(mb, 2))

            for method_name, ratio in compression_methods.items():
                curves_mb[method_name].append(round(mb * ratio, 2))

        savings = {}
        for method_name, ratio in compression_methods.items():
            savings[method_name] = f"{(1.0 - ratio) * 100:.1f}%"

        return {
            "seq_lens": seq_lens,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "bytes_per_elem": bytes_per_elem,
            "baseline_mb": baseline_mb,
            "baseline_gb": baseline_gb,
            "compression_curves_mb": curves_mb,
            "memory_savings_percent": savings,
        }

    def plot_memory_scaling(
        self,
        seq_lens: List[int],
        memory_curves_mb: Dict[str, List[float]],
        title: str = "KV Cache Memory Scaling vs Sequence Length",
        save: bool = True,
    ) -> Optional[str]:
        """Plots KV cache memory scaling curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        palette = sns.color_palette("tab10", len(memory_curves_mb))

        for (name, values), color in zip(memory_curves_mb.items(), palette):
            is_baseline = "Baseline" in name
            linewidth = 3.0 if is_baseline else 2.0
            linestyle = "-" if is_baseline else "--"
            ax.plot(seq_lens, values, label=name, color=color, linewidth=linewidth, linestyle=linestyle, marker="o")

        ax.set_xlabel("Sequence Length (Tokens)")
        ax.set_ylabel("KV Cache Memory (MB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

        filename = "kv_cache_scaling_comparison.png"
        return self._save_figure(fig, filename) if save else None

    def compute_delta_kv_heatmap(
        self,
        k_cache_t1: Union[np.ndarray, torch.Tensor],
        k_cache_t2: Union[np.ndarray, torch.Tensor],
        layer_idx: int = 0,
        head_idx: int = 0,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Delta-KV Temporal Difference:
        Computes Delta = |K_t2 - K_t1| across token positions and feature dimensions.
        """
        k1 = _to_numpy(k_cache_t1)
        k2 = _to_numpy(k_cache_t2)

        # Ensure matching shapes
        min_seq = min(k1.shape[0], k2.shape[0])
        diff = np.abs(k2[:min_seq] - k1[:min_seq])  # [seq_len, head_dim] or [seq_len, seq_len]

        mean_delta_per_token = np.mean(diff, axis=-1)
        static_threshold = np.percentile(mean_delta_per_token, 20)
        dynamic_threshold = np.percentile(mean_delta_per_token, 80)

        static_tokens = np.where(mean_delta_per_token <= static_threshold)[0].tolist()
        dynamic_tokens = np.where(mean_delta_per_token >= dynamic_threshold)[0].tolist()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(diff, cmap="magma", ax=ax, cbar_kws={"label": "Temporal Delta |K_t2 - K_t1|"})
        ax.set_title(f"Delta-KV Temporal Difference (Layer {layer_idx}, Head {head_idx})")
        ax.set_xlabel("Feature Dimension")
        ax.set_ylabel("Token Position")
        plt.tight_layout()

        filename = f"delta_kv_heatmap_layer{layer_idx}_head{head_idx}.png"
        saved_path = self._save_figure(fig, filename) if save else None

        return {
            "mean_delta": float(np.mean(diff)),
            "max_delta": float(np.max(diff)),
            "static_token_indices": static_tokens,
            "dynamic_token_indices": dynamic_tokens,
            "file_path": saved_path,
            "shape": list(diff.shape),
        }

    def detect_attention_sinks(
        self,
        attention_mass_or_scores: Union[np.ndarray, torch.Tensor],
        threshold_multiplier: float = 2.0,
        top_k_sinks: int = 4,
    ) -> Dict[str, Any]:
        """
        Detects attention sink tokens that absorb disproportionate attention mass.
        sink_score[token_i] = sum_{layers, heads} attention_weight[:, token_i]
        """
        attn = _to_numpy(attention_mass_or_scores)
        if attn.ndim == 3:
            # [heads, q_len, k_len] -> sum over heads and queries
            token_scores = np.sum(attn, axis=(0, 1))
        elif attn.ndim == 2:
            # [q_len, k_len] -> sum over queries
            token_scores = np.sum(attn, axis=0)
        else:
            token_scores = attn

        mean_score = float(np.mean(token_scores))
        std_score = float(np.std(token_scores))
        threshold = mean_score * threshold_multiplier

        sink_mask = token_scores >= threshold
        sink_indices = np.where(sink_mask)[0].tolist()

        # Top sink tokens
        top_indices = np.argsort(token_scores)[::-1][:top_k_sinks].tolist()
        total_mass = float(np.sum(token_scores)) + 1e-9
        sink_mass_sum = float(np.sum(token_scores[sink_indices])) if sink_indices else 0.0

        return {
            "sink_scores": token_scores.tolist(),
            "sink_indices": sink_indices,
            "top_sink_candidates": top_indices,
            "mean_score": mean_score,
            "std_score": std_score,
            "sink_threshold": threshold,
            "sink_mass_percentage": float((sink_mass_sum / total_mass) * 100.0),
        }

    def track_compression_ratios(
        self,
        original_kv: Union[np.ndarray, torch.Tensor],
        compressed_kv: Union[np.ndarray, torch.Tensor],
        bits_original: int = 16,
        bits_compressed: int = 4,
    ) -> Dict[str, Any]:
        """
        Tracks compression ratios, parameter reduction, and reconstruction quality.
        """
        orig = _to_numpy(original_kv)
        comp = _to_numpy(compressed_kv)

        orig_elements = orig.size
        comp_elements = comp.size

        orig_bits = orig_elements * bits_original
        comp_bits = comp_elements * bits_compressed
        compression_ratio = float(comp_bits / orig_bits) if orig_bits > 0 else 1.0

        # Frobenius norm error if same shape
        if orig.shape == comp.shape:
            frobenius_err = float(np.linalg.norm(orig - comp) / (np.linalg.norm(orig) + 1e-9))
        else:
            frobenius_err = None

        return {
            "original_elements": int(orig_elements),
            "compressed_elements": int(comp_elements),
            "original_size_kb": float(orig_bits / (8 * 1024)),
            "compressed_size_kb": float(comp_bits / (8 * 1024)),
            "compression_ratio": compression_ratio,
            "memory_saved_percent": float((1.0 - compression_ratio) * 100.0),
            "reconstruction_error": frobenius_err,
        }


# =============================================================================
# 3. WEIGHT MATRIX DECOMPOSER
# =============================================================================

@VISUALIZER_REGISTRY.register("weight_matrix_decomposer")
class WeightMatrixDecomposer(BaseVisualizer):
    """
    Weight Matrix Decomposer — Low-Rank Factorization Studio.
    Performs real-time SVD, CUR, and Robust PCA (RPCA) decompositions.
    """

    def __init__(self, output_dir: str = "visualizations/weight_decomposer") -> None:
        super().__init__(output_dir)

    def generate(self, W: Union[np.ndarray, torch.Tensor], rank: int = 32, **kwargs) -> Dict[str, Any]:
        """Default generation pipeline for weight matrix decomposition."""
        W_np = _to_numpy(W)
        svd_res = self.decompose_svd(W_np, rank=rank)
        cur_res = self.decompose_cur(W_np, rank=rank)
        spectrum_plot = self.plot_spectrum_and_energy(svd_res["singular_values"], rank=rank, save=kwargs.get("save", True))
        diff_plot = self.plot_matrix_reconstruction_diff(W_np, svd_res["reconstructed_W"], save=kwargs.get("save", True))
        return {
            "svd": svd_res,
            "cur": cur_res,
            "spectrum_plot": spectrum_plot,
            "diff_plot": diff_plot,
        }

    def decompose_svd(
        self,
        W: Union[np.ndarray, torch.Tensor],
        rank: Optional[int] = None,
        energy_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        SVD Decomposition: W = U Sigma V^T
        Returns truncated singular vectors, cumulative energy, reconstruction error.
        """
        mat = _to_numpy(W)
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        total_energy = np.sum(s ** 2)
        cum_energy = np.cumsum(s ** 2) / (total_energy + 1e-12)

        max_rank = len(s)
        if energy_threshold is not None:
            # Find first rank reaching energy threshold
            k = int(np.searchsorted(cum_energy, energy_threshold) + 1)
            k = min(max_rank, max(1, k))
        elif rank is not None:
            k = min(max_rank, max(1, rank))
        else:
            k = min(32, max_rank)

        # Truncate
        U_k = U[:, :k]
        s_k = s[:k]
        Vt_k = Vt[:k, :]
        W_k = U_k @ np.diag(s_k) @ Vt_k

        frob_err = float(np.linalg.norm(mat - W_k) / (np.linalg.norm(mat) + 1e-9))
        m, n = mat.shape
        orig_params = m * n
        factor_params = m * k + k + k * n
        param_reduction = float(1.0 - (factor_params / orig_params))

        return {
            "rank": k,
            "max_rank": max_rank,
            "singular_values": s.tolist(),
            "singular_values_truncated": s_k.tolist(),
            "cumulative_energy": cum_energy.tolist(),
            "energy_retained": float(cum_energy[k - 1]),
            "reconstruction_error": frob_err,
            "param_reduction_ratio": param_reduction,
            "reconstructed_W": W_k,
            "shape": list(mat.shape),
        }

    def decompose_cur(
        self,
        W: Union[np.ndarray, torch.Tensor],
        rank: Optional[int] = None,
        num_cols: Optional[int] = None,
        num_rows: Optional[int] = None,
        method: str = "leverage",
    ) -> Dict[str, Any]:
        """
        CUR Decomposition: W approx C @ U @ R
        Selects columns C and rows R via statistical leverage scores.
        """
        mat = _to_numpy(W)
        m, n = mat.shape
        k = rank or min(16, min(m, n))
        c = num_cols or min(k * 2, n)
        r = num_rows or min(k * 2, m)

        # SVD for leverage scores
        U, s, Vt = np.linalg.svd(mat, full_matrices=False)
        k_lev = min(k, len(s))

        # Normalized leverage scores
        row_leverage = np.sum(U[:, :k_lev] ** 2, axis=1) / k_lev
        col_leverage = np.sum(Vt[:k_lev, :] ** 2, axis=0) / k_lev

        if method == "leverage":
            row_probs = row_leverage / np.sum(row_leverage)
            col_probs = col_leverage / np.sum(col_leverage)
            selected_rows = np.argsort(row_probs)[::-1][:r]
            selected_cols = np.argsort(col_probs)[::-1][:c]
        else:
            # Uniform or norm-based
            col_norms = np.linalg.norm(mat, axis=0)
            row_norms = np.linalg.norm(mat, axis=1)
            selected_cols = np.argsort(col_norms)[::-1][:c]
            selected_rows = np.argsort(row_norms)[::-1][:r]

        selected_rows = np.sort(selected_rows)
        selected_cols = np.sort(selected_cols)

        C = mat[:, selected_cols]
        R = mat[selected_rows, :]

        # Middle matrix U = pinv(C) @ W @ pinv(R)
        C_pinv = np.linalg.pinv(C)
        R_pinv = np.linalg.pinv(R)
        U_mid = C_pinv @ mat @ R_pinv

        W_cur = C @ U_mid @ R
        frob_err = float(np.linalg.norm(mat - W_cur) / (np.linalg.norm(mat) + 1e-9))

        return {
            "rank": k,
            "num_cols": c,
            "num_rows": r,
            "selected_rows": selected_rows.tolist(),
            "selected_cols": selected_cols.tolist(),
            "row_leverage_scores": row_leverage.tolist(),
            "col_leverage_scores": col_leverage.tolist(),
            "reconstruction_error": frob_err,
            "reconstructed_W": W_cur,
            "C_shape": list(C.shape),
            "U_shape": list(U_mid.shape),
            "R_shape": list(R.shape),
        }

    def decompose_rpca(
        self,
        W: Union[np.ndarray, torch.Tensor],
        lambda_param: Optional[float] = None,
        max_iter: int = 100,
        tol: float = 1e-7,
    ) -> Dict[str, Any]:
        """
        Robust PCA (Inexact ALM): W = L + S (Low-rank + Sparse).
        """
        mat = _to_numpy(W)
        m, n = mat.shape
        lam = lambda_param or (1.0 / math.sqrt(max(m, n)))

        # Initialize
        Y = mat / max(np.linalg.norm(mat, 2), np.max(np.abs(mat)) / lam)
        L = np.zeros_like(mat)
        S = np.zeros_like(mat)
        mu = 1.25 / np.linalg.norm(mat, 2)
        mu_bar = mu * 1e7
        rho = 1.5
        d_norm = np.linalg.norm(mat, "fro")

        iter_count = 0
        converged = False

        for i in range(max_iter):
            iter_count = i + 1
            # Update L via Singular Value Thresholding
            temp_L = mat - S + (1.0 / mu) * Y
            U, s, Vt = np.linalg.svd(temp_L, full_matrices=False)
            s_thresh = np.maximum(s - (1.0 / mu), 0.0)
            rank_L = int(np.sum(s_thresh > 0))
            if rank_L > 0:
                L = (U[:, :rank_L] * s_thresh[:rank_L]) @ Vt[:rank_L, :]
            else:
                L = np.zeros_like(mat)

            # Update S via Soft Thresholding
            temp_S = mat - L + (1.0 / mu) * Y
            S = np.sign(temp_S) * np.maximum(np.abs(temp_S) - (lam / mu), 0.0)

            # Residual & Convergence check
            Z = mat - L - S
            Y = Y + mu * Z
            mu = min(mu * rho, mu_bar)

            err = np.linalg.norm(Z, "fro") / (d_norm + 1e-9)
            if err < tol:
                converged = True
                break

        U_L, s_L, _ = np.linalg.svd(L, full_matrices=False)
        final_rank = int(np.sum(s_L > 1e-5))
        sparsity_ratio = float(np.mean(np.abs(S) < 1e-5))

        return {
            "converged": converged,
            "iterations": iter_count,
            "estimated_rank_L": final_rank,
            "sparsity_ratio_S": sparsity_ratio,
            "reconstruction_error": float(np.linalg.norm(mat - (L + S)) / (d_norm + 1e-9)),
            "low_rank_L": L,
            "sparse_S": S,
        }

    def plot_spectrum_and_energy(
        self,
        singular_values: Union[np.ndarray, List[float]],
        rank: Optional[int] = None,
        energy_threshold: float = 0.95,
        title: str = "Singular Value Spectrum & Cumulative Energy Retention",
        save: bool = True,
    ) -> Optional[str]:
        """Plots 2-panel chart: Singular values + Cumulative energy curve."""
        s = np.array(singular_values)
        cum_energy = np.cumsum(s ** 2) / (np.sum(s ** 2) + 1e-12)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(s, "o-", color="#2980B9", markersize=3, linewidth=1.5)
        ax1.set_yscale("log")
        ax1.set_xlabel("Singular Value Index")
        ax1.set_ylabel("Singular Value (Log Scale)")
        ax1.set_title("Singular Value Spectrum")
        ax1.grid(True, alpha=0.3)
        if rank is not None:
            ax1.axvline(x=rank, color="#E74C3C", linestyle="--", label=f"Rank {rank}")
            ax1.legend()

        ax2.plot(cum_energy, "o-", color="#27AE60", markersize=3, linewidth=1.5)
        ax2.axhline(y=energy_threshold, color="#E67E22", linestyle="--", label=f"{energy_threshold*100:.0f}% Energy Target")
        if rank is not None and rank <= len(cum_energy):
            ax2.axvline(x=rank, color="#E74C3C", linestyle="--", label=f"Rank {rank} ({cum_energy[rank-1]*100:.1f}%)")
        ax2.set_xlabel("Rank (Components Kept)")
        ax2.set_ylabel("Cumulative Energy Ratio")
        ax2.set_title("Energy Retention Curve")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.suptitle(title)
        plt.tight_layout()
        filename = "svd_spectrum_energy_profile.png"
        return self._save_figure(fig, filename) if save else None

    def plot_matrix_reconstruction_diff(
        self,
        W_orig: Union[np.ndarray, torch.Tensor],
        W_approx: Union[np.ndarray, torch.Tensor],
        title: str = "Weight Matrix Reconstruction & Error Map",
        save: bool = True,
    ) -> Optional[str]:
        """Plots 3-panel comparison: Original, Reconstructed, Absolute Difference."""
        w1 = _to_numpy(W_orig)
        w2 = _to_numpy(W_approx)
        diff = np.abs(w1 - w2)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        sns.heatmap(w1, cmap="coolwarm", ax=axes[0], cbar=False)
        axes[0].set_title("Original Matrix W")

        sns.heatmap(w2, cmap="coolwarm", ax=axes[1], cbar=False)
        axes[1].set_title("Low-Rank Reconstructed W'")

        sns.heatmap(diff, cmap="Reds", ax=axes[2], cbar_kws={"label": "Absolute Error"})
        axes[2].set_title("Error Map |W - W'|")

        plt.suptitle(title)
        plt.tight_layout()
        filename = "matrix_reconstruction_diff.png"
        return self._save_figure(fig, filename) if save else None


# =============================================================================
# 4. LoRA ADAPTER STUDIO
# =============================================================================

@VISUALIZER_REGISTRY.register("lora_adapter_studio")
class LoRAAdapterStudio(BaseVisualizer):
    """
    LoRA Adapter Studio — Rank Spectrum & Low-Rank Gradient Dynamics.
    Analyzes adapter matrices A (r x d_in) and B (d_out x r), delta-W updates, and optimization trajectories.
    """

    def __init__(self, output_dir: str = "visualizations/lora_studio") -> None:
        super().__init__(output_dir)

    def generate(self, A: Union[np.ndarray, torch.Tensor], B: Union[np.ndarray, torch.Tensor], alpha: float = 1.0, **kwargs) -> Dict[str, Any]:
        """Default generation pipeline for LoRA analysis."""
        spec_info = self.analyze_rank_spectrum(A, B, alpha=alpha)
        plot_path = self.plot_lora_spectrum(spec_info["singular_values"], spec_info["delta_W"], save=kwargs.get("save", True))
        return {
            "spectrum": spec_info,
            "plot_path": plot_path,
        }

    def analyze_rank_spectrum(
        self,
        A: Union[np.ndarray, torch.Tensor],
        B: Union[np.ndarray, torch.Tensor],
        alpha: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Analyzes the rank spectrum of adapter matrices:
        Delta_W = (alpha / r) * B @ A
        """
        A_np = _to_numpy(A)
        B_np = _to_numpy(B)
        r = A_np.shape[0]

        scaling = float(alpha / r) if r > 0 else 1.0
        delta_W = scaling * (B_np @ A_np)

        U, s, Vt = np.linalg.svd(delta_W, full_matrices=False)
        total_energy = np.sum(s ** 2)
        cum_energy = np.cumsum(s ** 2) / (total_energy + 1e-12)

        # Effective rank via singular value entropy
        s_norm = s / (np.sum(s) + 1e-12)
        s_nonzero = s_norm[s_norm > 1e-12]
        eff_rank = float(np.exp(-np.sum(s_nonzero * np.log(s_nonzero))))

        # Condition number
        cond_num = float(s[0] / (s[-1] + 1e-12)) if len(s) > 0 else 1.0

        return {
            "rank": r,
            "alpha": alpha,
            "scaling": scaling,
            "delta_W": delta_W,
            "singular_values": s.tolist(),
            "singular_values_adapter": s[:r].tolist(),
            "cumulative_energy": cum_energy.tolist(),
            "effective_rank": eff_rank,
            "condition_number": cond_num,
            "norm_delta_w": float(np.linalg.norm(delta_W, "fro")),
            "d_out": B_np.shape[0],
            "d_in": A_np.shape[1],
        }

    def compute_delta_w_magnitudes(
        self,
        W_base: Union[np.ndarray, torch.Tensor],
        A: Union[np.ndarray, torch.Tensor],
        B: Union[np.ndarray, torch.Tensor],
        alpha: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Inspects adapter magnitude relative to base weights:
        Relative magnitude = ||Delta_W||_F / ||W_base||_F
        """
        W_b = _to_numpy(W_base)
        A_np = _to_numpy(A)
        B_np = _to_numpy(B)
        r = A_np.shape[0]
        scaling = float(alpha / r) if r > 0 else 1.0
        delta_W = scaling * (B_np @ A_np)

        norm_base = float(np.linalg.norm(W_b, "fro"))
        norm_adapter = float(np.linalg.norm(delta_W, "fro"))
        rel_mag = float(norm_adapter / (norm_base + 1e-9))

        # Row-wise and column-wise update intensities
        row_intensities = np.linalg.norm(delta_W, axis=1).tolist()
        col_intensities = np.linalg.norm(delta_W, axis=0).tolist()

        # Cosine similarity between base and delta
        flat_w = W_b.flatten()
        flat_dw = delta_W.flatten()
        cos_sim = float(np.dot(flat_w, flat_dw) / ((np.linalg.norm(flat_w) * np.linalg.norm(flat_dw)) + 1e-12))

        return {
            "base_norm": norm_base,
            "adapter_norm": norm_adapter,
            "relative_magnitude": rel_mag,
            "cosine_similarity": cos_sim,
            "row_intensities": row_intensities,
            "col_intensities": col_intensities,
        }

    def simulate_low_rank_gradient_dynamics(
        self,
        W_base: Union[np.ndarray, torch.Tensor],
        A: Union[np.ndarray, torch.Tensor],
        B: Union[np.ndarray, torch.Tensor],
        grad_output: Union[np.ndarray, torch.Tensor],
        x: Union[np.ndarray, torch.Tensor],
        lr: float = 1e-4,
        alpha: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Simulates low-rank gradient updates:
        dL/dB = (alpha/r) * (grad_out) @ (A x)^T
        dL/dA = (alpha/r) * B^T @ (grad_out) @ x^T
        """
        W_b = _to_numpy(W_base)
        A_np = _to_numpy(A)
        B_np = _to_numpy(B)
        dY = _to_numpy(grad_output)  # [d_out, seq_len] or [d_out, 1]
        X_in = _to_numpy(x)          # [d_in, seq_len] or [d_in, 1]

        if dY.ndim == 1:
            dY = dY.reshape(-1, 1)
        if X_in.ndim == 1:
            X_in = X_in.reshape(-1, 1)

        r = A_np.shape[0]
        scaling = float(alpha / r) if r > 0 else 1.0

        # Intermediate activation h = A @ x
        h = A_np @ X_in  # [r, seq_len]

        # Gradients
        grad_B = scaling * (dY @ h.T)            # [d_out, r]
        grad_A = scaling * ((B_np.T @ dY) @ X_in.T) # [r, d_in]
        full_grad_W = dY @ X_in.T                # [d_out, d_in]

        # Projected low-rank gradient in weight space
        grad_delta_W = scaling * (grad_B @ A_np + B_np @ grad_A)

        # Gradient subspace alignment
        flat_proj = grad_delta_W.flatten()
        flat_full = full_grad_W.flatten()
        subspace_alignment = float(np.dot(flat_proj, flat_full) / ((np.linalg.norm(flat_proj) * np.linalg.norm(flat_full)) + 1e-12))

        # Perform 1-step simulated SGD update
        A_next = A_np - lr * grad_A
        B_next = B_np - lr * grad_B
        delta_W_next = scaling * (B_next @ A_next)

        return {
            "grad_A_norm": float(np.linalg.norm(grad_A, "fro")),
            "grad_B_norm": float(np.linalg.norm(grad_B, "fro")),
            "full_grad_norm": float(np.linalg.norm(full_grad_W, "fro")),
            "subspace_alignment": subspace_alignment,
            "updated_delta_norm": float(np.linalg.norm(delta_W_next, "fro")),
            "learning_rate": lr,
        }

    def compare_initializations(
        self,
        d_out: int,
        d_in: int,
        rank: int,
        W_base: Optional[Union[np.ndarray, torch.Tensor]] = None,
        alpha: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compares Standard Zero-Init (Kaiming A, Zero B) vs SVD-Init (Top-r SVD components of W_base).
        """
        if W_base is None:
            # Generate synthetic structured base weight
            np.random.seed(42)
            W_b = np.random.randn(d_out, d_in).astype(np.float32) / math.sqrt(d_in)
        else:
            W_b = _to_numpy(W_base)

        # 1. Standard Initialization
        A_std = np.random.randn(rank, d_in).astype(np.float32) / math.sqrt(d_in)
        B_std = np.zeros((d_out, rank), dtype=np.float32)
        delta_std = (alpha / rank) * (B_std @ A_std)
        std_error = float(np.linalg.norm(W_b - (W_b + delta_std)) / (np.linalg.norm(W_b) + 1e-9))

        # 2. SVD Initialization
        U, s, Vt = np.linalg.svd(W_b, full_matrices=False)
        r = min(rank, len(s))
        sqrt_s = np.sqrt(s[:r])
        B_svd = U[:, :r] * sqrt_s
        A_svd = sqrt_s[:, np.newaxis] * Vt[:r, :]
        delta_svd = (alpha / rank) * (B_svd @ A_svd)
        svd_energy_captured = float(np.sum(s[:r] ** 2) / (np.sum(s ** 2) + 1e-12))
        svd_reconstruction_err = float(np.linalg.norm(W_b - delta_svd) / (np.linalg.norm(W_b) + 1e-9))

        return {
            "rank": rank,
            "d_out": d_out,
            "d_in": d_in,
            "standard_init": {
                "norm_delta_w": float(np.linalg.norm(delta_std)),
                "norm_A": float(np.linalg.norm(A_std)),
                "norm_B": float(np.linalg.norm(B_std)),
                "initial_perturbation": std_error,
            },
            "svd_init": {
                "norm_delta_w": float(np.linalg.norm(delta_svd)),
                "norm_A": float(np.linalg.norm(A_svd)),
                "norm_B": float(np.linalg.norm(B_svd)),
                "energy_captured": svd_energy_captured,
                "reconstruction_error": svd_reconstruction_err,
            },
        }

    def plot_lora_spectrum(
        self,
        singular_values: Union[np.ndarray, List[float]],
        delta_w: Optional[Union[np.ndarray, torch.Tensor]] = None,
        title: str = "LoRA Adapter Singular Spectrum & Delta-W Matrix",
        save: bool = True,
    ) -> Optional[str]:
        """Plots 2-panel chart: Adapter singular value distribution + Delta-W heatmap."""
        s = np.array(singular_values)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(range(1, len(s) + 1), s, color="#8E44AD", alpha=0.8)
        axes[0].set_xlabel("Singular Value Index")
        axes[0].set_ylabel("Magnitude")
        axes[0].set_title("LoRA Delta-W Spectrum")
        axes[0].grid(True, alpha=0.3)

        if delta_w is not None:
            dw_np = _to_numpy(delta_w)
            sns.heatmap(dw_np, cmap="coolwarm", ax=axes[1], cbar_kws={"label": "Weight Update Magnitude"})
            axes[1].set_title(f"Delta-W Heatmap ({dw_np.shape[0]}x{dw_np.shape[1]})")
        else:
            axes[1].text(0.5, 0.5, "No Delta-W Matrix Provided", ha="center", va="center")

        plt.suptitle(title)
        plt.tight_layout()
        filename = "lora_adapter_spectrum_profile.png"
        return self._save_figure(fig, filename) if save else None
