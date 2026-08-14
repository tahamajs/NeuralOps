"""
Representation & Internal State Studios for Transformer Analysis.

This module implements the core computational engines and visualizers for:
1. EmbeddingSpaceExplorer: 3D token embedding projections (PCA, t-SNE, UMAP), cosine similarity matrices,
   semantic axis discovery via ICA/PCA, anisotropy metrics, and linear probe training.
2. PositionalEncodingStudio: RoPE rotation matrix generator, sinusoidal wavelength heatmaps,
   frequency spectrum analysis, and relative-position similarity matrices.
3. ResidualStreamTracer: Layer-wise hidden state norm tracking, vector update arithmetic,
   2D/3D vector dynamics sandbox, and Logit Lens / Logit Prisms linear decomposition.
4. LayerNormDynamics: Mean/variance scaling, gain/bias parameter drift, hypersphere projection,
   and Pre-LN vs Post-LN stability analysis.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import VISUALIZER_REGISTRY, BaseVisualizer
from .utils import create_custom_colormap, set_plot_style


# =====================================================================
# Helper Utilities
# =====================================================================

def _to_numpy(tensor_or_array: Union[torch.Tensor, np.ndarray, List[float]]) -> np.ndarray:
    """Converts a PyTorch tensor, numpy array, or list to a float32 numpy ndarray."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().to(torch.float32).numpy()
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return np.array(tensor_or_array, dtype=np.float32)


def _safe_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """L2-normalizes vectors along the specified axis with numerical stability."""
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, eps)


# =====================================================================
# 1. Embedding Space Explorer
# =====================================================================

@dataclass
class EmbeddingProjectionResult:
    """Data container for 2D/3D embedding dimensionality reduction."""
    coordinates: np.ndarray  # Shape: (N, n_components)
    tokens: List[str]
    method: str
    explained_variance_ratio: Optional[np.ndarray] = None
    cluster_labels: Optional[np.ndarray] = None
    anisotropy_score: float = 0.0
    average_l2_norm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coordinates": self.coordinates.tolist(),
            "tokens": self.tokens,
            "method": self.method,
            "explained_variance_ratio": (
                self.explained_variance_ratio.tolist()
                if self.explained_variance_ratio is not None
                else None
            ),
            "cluster_labels": (
                self.cluster_labels.tolist()
                if self.cluster_labels is not None
                else None
            ),
            "anisotropy_score": float(self.anisotropy_score),
            "average_l2_norm": float(self.average_l2_norm),
        }


@dataclass
class SemanticAxisResult:
    """Discovered or user-defined semantic axis in embedding space."""
    name: str
    axis_vector: np.ndarray  # Shape: (d_model,)
    token_projections: Dict[str, float]  # token -> projection score
    top_positive_tokens: List[Tuple[str, float]]
    top_negative_tokens: List[Tuple[str, float]]
    explained_variance: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "axis_vector": self.axis_vector.tolist(),
            "token_projections": {k: float(v) for k, v in self.token_projections.items()},
            "top_positive_tokens": [(t, float(s)) for t, s in self.top_positive_tokens],
            "top_negative_tokens": [(t, float(s)) for t, s in self.top_negative_tokens],
            "explained_variance": float(self.explained_variance),
        }


@VISUALIZER_REGISTRY.register("embedding_space_explorer")
class EmbeddingSpaceExplorer(BaseVisualizer):
    """
    Explorer and diagnostic suite for token embedding spaces.
    Provides 3D projections (PCA, t-SNE, UMAP), cosine similarity analysis,
    semantic axis discovery (ICA / vector arithmetic), anisotropy metrics, and linear probe training.
    """

    def __init__(self, output_dir: str = "visualizations/embeddings") -> None:
        super().__init__(output_dir)
        self.cmap = create_custom_colormap()
        set_plot_style()

    def compute_cosine_similarity_matrix(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
    ) -> np.ndarray:
        """
        Computes pairwise cosine similarity matrix S_{ij} = cos(e_i, e_j).

        Args:
            embeddings: Token embedding matrix of shape (N, d_model).

        Returns:
            Cosine similarity matrix of shape (N, N).
        """
        emb = _to_numpy(embeddings)
        normed = _safe_normalize(emb, axis=-1)
        sim_matrix = np.matmul(normed, normed.T)
        return np.clip(sim_matrix, -1.0, 1.0)

    def compute_anisotropy_metrics(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, float]:
        """
        Measures anisotropy and directional concentration in the embedding space.
        Anisotropic embedding spaces concentrate token vectors in a narrow cone.

        Returns:
            Dictionary containing:
                - average_cosine_similarity: Mean pairwise similarity (high = anisotropic).
                - anisotropy_score: 1 - isotropy_index (where isotropy_index = min_singular / max_singular).
                - avg_l2_norm: Average L2 norm of embeddings.
                - l2_norm_std: Standard deviation of L2 norms across tokens.
        """
        emb = _to_numpy(embeddings)
        normed = _safe_normalize(emb, axis=-1)
        sim_matrix = np.matmul(normed, normed.T)
        n = sim_matrix.shape[0]

        # Exclude self-similarity diagonal
        triu_idx = np.triu_indices(n, k=1)
        if len(triu_idx[0]) > 0:
            avg_cos_sim = float(np.mean(sim_matrix[triu_idx]))
        else:
            avg_cos_sim = 1.0

        # SVD of centered normalized embeddings
        centered = normed - np.mean(normed, axis=0, keepdims=True)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        if len(s) > 0 and s[0] > 1e-8:
            isotropy_index = float(s[-1] / s[0])
        else:
            isotropy_index = 0.0

        l2_norms = np.linalg.norm(emb, axis=-1)
        return {
            "average_cosine_similarity": avg_cos_sim,
            "isotropy_index": isotropy_index,
            "anisotropy_score": 1.0 - isotropy_index,
            "avg_l2_norm": float(np.mean(l2_norms)),
            "l2_norm_std": float(np.std(l2_norms)),
        }

    def project_embeddings(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        tokens: List[str],
        method: str = "pca",
        n_components: int = 3,
        n_clusters: Optional[int] = None,
        random_state: int = 42,
    ) -> EmbeddingProjectionResult:
        """
        Projects high-dimensional embeddings to 2D or 3D using PCA, t-SNE, or UMAP.

        Args:
            embeddings: Array of shape (N, d_model).
            tokens: List of token strings.
            method: 'pca', 'tsne', or 'umap'.
            n_components: Target projection dimensionality (2 or 3).
            n_clusters: Optional number of k-means clusters.
            random_state: Random seed for reproducibility.

        Returns:
            EmbeddingProjectionResult containing coordinates, variance ratios, and cluster labels.
        """
        emb = _to_numpy(embeddings)
        n_samples, d_model = emb.shape
        n_components = min(n_components, d_model, n_samples)

        explained_variance_ratio = None
        method_lower = method.lower()

        if method_lower == "pca":
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_components, random_state=random_state)
            coords = pca.fit_transform(emb)
            explained_variance_ratio = pca.explained_variance_ratio_

        elif method_lower == "tsne":
            from sklearn.manifold import TSNE
            perplexity = min(30.0, max(1.0, float(n_samples - 1) / 3.0))
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
                init="pca",
                learning_rate="auto",
                max_iter=300,
            )
            coords = tsne.fit_transform(emb)

        elif method_lower == "umap":
            try:
                import umap
                reducer = umap.UMAP(
                    n_components=n_components,
                    random_state=random_state,
                    n_neighbors=min(15, n_samples - 1) if n_samples > 1 else 1,
                )
                coords = reducer.fit_transform(emb)
            except ImportError:
                # Fallback to PCA if umap is not installed
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_components, random_state=random_state)
                coords = pca.fit_transform(emb)
                explained_variance_ratio = pca.explained_variance_ratio_
                method_lower = "pca (umap-fallback)"
        else:
            raise ValueError(f"Unknown projection method: {method}. Choose 'pca', 'tsne', or 'umap'.")

        cluster_labels = None
        if n_clusters is not None and n_clusters > 1 and n_samples >= n_clusters:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(coords)

        metrics = self.compute_anisotropy_metrics(emb)

        return EmbeddingProjectionResult(
            coordinates=coords,
            tokens=tokens,
            method=method_lower,
            explained_variance_ratio=explained_variance_ratio,
            cluster_labels=cluster_labels,
            anisotropy_score=metrics["anisotropy_score"],
            average_l2_norm=metrics["avg_l2_norm"],
        )

    def discover_semantic_axes(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        tokens: List[str],
        n_axes: int = 3,
        method: str = "ica",
        random_state: int = 42,
    ) -> List[SemanticAxisResult]:
        """
        Discovers intrinsic interpretable semantic axes via ICA (FastICA) or PCA decomposition.

        Args:
            embeddings: Token embedding matrix of shape (N, d_model).
            tokens: List of token strings.
            n_axes: Number of semantic axes to extract.
            method: 'ica' or 'pca'.
            random_state: Random seed.

        Returns:
            List of SemanticAxisResult items with token projection rankings.
        """
        emb = _to_numpy(embeddings)
        n_samples, d_model = emb.shape
        n_axes = min(n_axes, d_model, n_samples)

        if method.lower() == "ica":
            from sklearn.decomposition import FastICA
            ica = FastICA(n_components=n_axes, random_state=random_state, max_iter=500)
            projections = ica.fit_transform(emb)
            components = ica.components_
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=n_axes, random_state=random_state)
            projections = pca.fit_transform(emb)
            components = pca.components_

        results = []
        for i in range(n_axes):
            axis_vec = components[i]
            scores = {tokens[j]: float(projections[j, i]) for j in range(n_samples)}
            sorted_tokens = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            top_pos = sorted_tokens[:5]
            top_neg = sorted_tokens[-5:][::-1]

            variance = float(np.var(projections[:, i]))
            axis_name = f"Semantic Axis {i+1} ({top_pos[0][0]} vs {top_neg[0][0]})"

            results.append(
                SemanticAxisResult(
                    name=axis_name,
                    axis_vector=axis_vec,
                    token_projections=scores,
                    top_positive_tokens=top_pos,
                    top_negative_tokens=top_neg,
                    explained_variance=variance,
                )
            )
        return results

    def compute_directional_semantic_axis(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        tokens: List[str],
        positive_tokens: List[str],
        negative_tokens: List[str],
        axis_name: str = "Custom Axis",
    ) -> SemanticAxisResult:
        """
        Constructs a user-defined semantic axis via vector arithmetic:
        axis_vector = mean(positive_tokens) - mean(negative_tokens).
        """
        emb = _to_numpy(embeddings)
        token_to_idx = {t: i for i, t in enumerate(tokens)}

        pos_indices = [token_to_idx[t] for t in positive_tokens if t in token_to_idx]
        neg_indices = [token_to_idx[t] for t in negative_tokens if t in token_to_idx]

        if not pos_indices or not neg_indices:
            raise ValueError("Positive and negative token sets must each have at least one token present in vocabulary.")

        pos_vec = np.mean(emb[pos_indices], axis=0)
        neg_vec = np.mean(emb[neg_indices], axis=0)
        axis_vector = pos_vec - neg_vec
        norm = np.linalg.norm(axis_vector)
        if norm > 1e-8:
            axis_vector = axis_vector / norm

        # Project all tokens onto the axis
        proj_scores = np.dot(emb, axis_vector)
        scores = {tokens[i]: float(proj_scores[i]) for i in range(len(tokens))}
        sorted_tokens = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        return SemanticAxisResult(
            name=axis_name,
            axis_vector=axis_vector,
            token_projections=scores,
            top_positive_tokens=sorted_tokens[:5],
            top_negative_tokens=sorted_tokens[-5:][::-1],
            explained_variance=float(np.var(proj_scores)),
        )

    def train_linear_probe(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        labels: Union[List[int], np.ndarray],
        c_reg: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Trains a linear probe (L2-regularized logistic regression) to assess semantic linear separability.

        Args:
            embeddings: Token embedding matrix of shape (N, d_model).
            labels: Binary or multi-class labels for each token.
            c_reg: Inverse regularization parameter C.

        Returns:
            Dictionary with accuracy, precision, recall, F1, and hyperplane weights.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        emb = _to_numpy(embeddings)
        y = np.array(labels)

        clf = LogisticRegression(C=c_reg, max_iter=500, random_state=42)
        clf.fit(emb, y)
        preds = clf.predict(emb)

        acc = float(accuracy_score(y, preds))
        prec = float(precision_score(y, preds, average="weighted", zero_division=0))
        rec = float(recall_score(y, preds, average="weighted", zero_division=0))
        f1 = float(f1_score(y, preds, average="weighted", zero_division=0))

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "coefficients": clf.coef_.tolist(),
            "intercept": clf.intercept_.tolist(),
            "classes": clf.classes_.tolist(),
        }

    def generate(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        tokens: List[str],
        save_name: str = "embedding_space_analysis.png",
    ) -> str:
        """Generates summary visualization of the embedding space."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Subplot 1: 2D PCA projection
        proj = self.project_embeddings(embeddings, tokens, method="pca", n_components=2)
        coords = proj.coordinates
        axes[0].scatter(coords[:, 0], coords[:, 1], c="teal", alpha=0.7, edgecolors="white", s=60)
        for i, txt in enumerate(tokens[:30]):  # annotate first 30 tokens
            axes[0].annotate(txt, (coords[i, 0], coords[i, 1]), fontsize=9, alpha=0.85)
        axes[0].set_title(f"Embedding Space (2D PCA Projection) - Anisotropy: {proj.anisotropy_score:.3f}")
        axes[0].set_xlabel("PC 1")
        axes[0].set_ylabel("PC 2")

        # Subplot 2: Cosine similarity heatmap
        sim_mat = self.compute_cosine_similarity_matrix(embeddings[:30])
        sns.heatmap(
            sim_mat,
            xticklabels=tokens[:30],
            yticklabels=tokens[:30],
            cmap=self.cmap,
            ax=axes[1],
            cbar_kws={"label": "Cosine Similarity"},
        )
        axes[1].set_title("Pairwise Cosine Similarity Submatrix")
        plt.tight_layout()
        return self._save_figure(fig, save_name)


# =====================================================================
# 2. Positional Encoding Studio
# =====================================================================

@dataclass
class RoPERotationState:
    """Mathematical rotation state for RoPE at a given position."""
    position: int
    dim_pair_idx: int
    base_frequency: float
    rotation_angle_rad: float
    rotation_angle_deg: float
    rotation_matrix: np.ndarray  # Shape: (2, 2)
    original_vector: np.ndarray  # Shape: (2,)
    rotated_vector: np.ndarray  # Shape: (2,)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position,
            "dim_pair_idx": self.dim_pair_idx,
            "base_frequency": float(self.base_frequency),
            "rotation_angle_rad": float(self.rotation_angle_rad),
            "rotation_angle_deg": float(self.rotation_angle_deg),
            "rotation_matrix": self.rotation_matrix.tolist(),
            "original_vector": self.original_vector.tolist(),
            "rotated_vector": self.rotated_vector.tolist(),
        }


@VISUALIZER_REGISTRY.register("positional_encoding_studio")
class PositionalEncodingStudio(BaseVisualizer):
    """
    Interactive laboratory for Rotary Position Embeddings (RoPE) and Sinusoidal Encodings.
    Provides RoPE rotation matrix generators, wavelength spectrum calculators,
    relative position similarity matrices, and Fourier frequency decompositions.
    """

    def __init__(self, output_dir: str = "visualizations/positional_encoding") -> None:
        super().__init__(output_dir)
        self.cmap = create_custom_colormap()
        set_plot_style()

    def generate_sinusoidal_table(
        self,
        max_len: int = 128,
        d_model: int = 64,
        base: float = 10000.0,
    ) -> np.ndarray:
        """
        Generates standard sinusoidal positional encoding matrix P of shape (max_len, d_model):
            P(pos, 2i)   = sin(pos / base^(2i / d_model))
            P(pos, 2i+1) = cos(pos / base^(2i / d_model))
        """
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        positions = np.arange(max_len, dtype=np.float32)[:, np.newaxis]
        dim_indices = np.arange(0, d_model, 2, dtype=np.float32)
        div_term = np.exp(dim_indices * -(math.log(base) / d_model))

        pe[:, 0::2] = np.sin(positions * div_term)
        pe[:, 1::2] = np.cos(positions * div_term)
        return pe

    def compute_rope_frequencies(
        self,
        d_model: int = 64,
        base: float = 10000.0,
    ) -> np.ndarray:
        r"""
        Computes base frequencies \theta_i = base^{-2i / d_model} for dimension pairs i = 0 ... d/2 - 1.
        """
        num_pairs = d_model // 2
        dim_indices = np.arange(num_pairs, dtype=np.float32)
        return 1.0 / (base ** (2.0 * dim_indices / float(d_model)))

    def generate_rope_rotation_matrix_2d(self, theta: float) -> np.ndarray:
        """
        Returns the 2x2 rotation matrix R(theta) = [[cos θ, -sin θ], [sin θ, cos θ]].
        """
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]], dtype=np.float32)

    def apply_rope(
        self,
        x: Union[torch.Tensor, np.ndarray],
        position: int,
        base: float = 10000.0,
    ) -> np.ndarray:
        """
        Applies Rotary Position Embedding (RoPE) to a d_model vector x at a given position.

        Args:
            x: Input vector of shape (d_model,) or batch/sequence of vectors.
            position: Integer token position p.
            base: Base frequency parameter (default 10000.0).

        Returns:
            RoPE-transformed vector of the same shape.
        """
        x_np = _to_numpy(x)
        orig_shape = x_np.shape
        d_model = orig_shape[-1]
        num_pairs = d_model // 2
        thetas = self.compute_rope_frequencies(d_model=d_model, base=base)

        x_flat = x_np.reshape(-1, d_model)
        out_flat = np.zeros_like(x_flat)

        for i in range(num_pairs):
            angle = float(position) * thetas[i]
            r = self.generate_rope_rotation_matrix_2d(angle)
            pair = x_flat[:, [2 * i, 2 * i + 1]]  # Shape: (B, 2)
            rotated = np.matmul(pair, r.T)
            out_flat[:, 2 * i] = rotated[:, 0]
            out_flat[:, 2 * i + 1] = rotated[:, 1]

        return out_flat.reshape(orig_shape)

    def get_rope_rotation_state(
        self,
        vector_2d: Union[List[float], np.ndarray],
        position: int,
        dim_pair_idx: int,
        d_model: int = 64,
        base: float = 10000.0,
    ) -> RoPERotationState:
        """Computes granular mathematical rotation state for a single dimension pair."""
        v = np.array(vector_2d, dtype=np.float32)[:2]
        thetas = self.compute_rope_frequencies(d_model=d_model, base=base)
        theta_i = thetas[dim_pair_idx]
        angle_rad = float(position) * float(theta_i)
        angle_deg = math.degrees(angle_rad) % 360.0

        r_matrix = self.generate_rope_rotation_matrix_2d(angle_rad)
        v_rot = np.matmul(r_matrix, v)

        return RoPERotationState(
            position=position,
            dim_pair_idx=dim_pair_idx,
            base_frequency=float(theta_i),
            rotation_angle_rad=angle_rad,
            rotation_angle_deg=angle_deg,
            rotation_matrix=r_matrix,
            original_vector=v,
            rotated_vector=v_rot,
        )

    def compute_position_similarity_matrix(
        self,
        encoding_type: str = "sinusoidal",
        max_len: int = 64,
        d_model: int = 64,
        base: float = 10000.0,
    ) -> np.ndarray:
        """
        Computes pairwise similarity matrix between positional vectors at all positions (i, j).

        Args:
            encoding_type: 'sinusoidal', 'rope', 'alibi', or 'learned'.
            max_len: Sequence length.
            d_model: Dimensionality.
            base: RoPE / sinusoidal base.

        Returns:
            Cosine similarity matrix of shape (max_len, max_len).
        """
        encoding_type = encoding_type.lower()
        if encoding_type == "sinusoidal":
            pe = self.generate_sinusoidal_table(max_len=max_len, d_model=d_model, base=base)
            normed = _safe_normalize(pe, axis=-1)
            return np.matmul(normed, normed.T)

        elif encoding_type == "rope":
            # For RoPE, test query-key dot product of unit vectors across positions
            thetas = self.compute_rope_frequencies(d_model=d_model, base=base)
            sim_mat = np.zeros((max_len, max_len), dtype=np.float32)
            # Dot product of rotated constant vector
            for i in range(max_len):
                for j in range(max_len):
                    diff = i - j
                    # Sum of cos(diff * theta_k) across all pairs
                    cos_terms = np.cos(diff * thetas)
                    sim_mat[i, j] = float(np.mean(cos_terms))
            return sim_mat

        elif encoding_type == "alibi":
            # ALiBi (Attention with Linear Biases): penalty m * (j - i)
            m = 2.0 ** (-8.0 / 8.0)  # Representative slope
            sim_mat = np.zeros((max_len, max_len), dtype=np.float32)
            for i in range(max_len):
                for j in range(max_len):
                    sim_mat[i, j] = max(0.0, 1.0 - m * abs(i - j) / 10.0)
            return sim_mat

        else:
            pe = self.generate_sinusoidal_table(max_len=max_len, d_model=d_model, base=base)
            normed = _safe_normalize(pe, axis=-1)
            return np.matmul(normed, normed.T)

    def analyze_frequency_spectrum(
        self,
        max_len: int = 128,
        d_model: int = 64,
        base: float = 10000.0,
    ) -> Dict[str, Any]:
        """
        Computes analytical wavelengths and Fourier power spectral density across positions.

        Returns:
            Dictionary containing wavelengths, frequencies, and FFT power spectrum per dimension.
        """
        num_pairs = d_model // 2
        frequencies = self.compute_rope_frequencies(d_model=d_model, base=base)
        wavelengths = (2.0 * math.pi) / frequencies

        # FFT of sinusoidal encodings
        pe = self.generate_sinusoidal_table(max_len=max_len, d_model=d_model, base=base)
        fft_vals = np.fft.rfft(pe, axis=0)
        power_spectrum = np.abs(fft_vals) ** 2

        return {
            "dim_pairs": list(range(num_pairs)),
            "frequencies": frequencies.tolist(),
            "wavelengths": wavelengths.tolist(),
            "power_spectrum": power_spectrum.tolist(),
        }

    def generate(
        self,
        max_len: int = 64,
        d_model: int = 64,
        save_name: str = "positional_encoding_analysis.png",
    ) -> str:
        """Generates comprehensive multi-panel figure for positional encodings."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 1. Sinusoidal heatmap
        pe = self.generate_sinusoidal_table(max_len=max_len, d_model=d_model)
        sns.heatmap(pe, cmap=self.cmap, ax=axes[0], cbar_kws={"label": "Encoding Value"})
        axes[0].set_title("Sinusoidal Positional Encoding Heatmap")
        axes[0].set_xlabel("Embedding Dimension")
        axes[0].set_ylabel("Token Position")

        # 2. RoPE relative similarity matrix
        rope_sim = self.compute_position_similarity_matrix(encoding_type="rope", max_len=max_len, d_model=d_model)
        sns.heatmap(rope_sim, cmap="coolwarm", ax=axes[1], cbar_kws={"label": "Relative Alignment"})
        axes[1].set_title("RoPE Relative Position Dot Product Matrix")
        axes[1].set_xlabel("Key Position (j)")
        axes[1].set_ylabel("Query Position (i)")

        # 3. Frequency Spectrum & Wavelengths
        spec = self.analyze_frequency_spectrum(max_len=max_len, d_model=d_model)
        axes[2].semilogy(spec["dim_pairs"], spec["wavelengths"], "o-", color="coral", lw=2)
        axes[2].set_title("Wavelength Scaling Across Dimension Pairs")
        axes[2].set_xlabel("Dimension Pair Index (i)")
        axes[2].set_ylabel("Wavelength (Tokens)")
        axes[2].grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        return self._save_figure(fig, save_name)


# =====================================================================
# 3. Residual Stream Tracer
# =====================================================================

@dataclass
class ResidualLayerTrace:
    """Trace data for a single transformer layer's residual stream."""
    layer_index: int
    hidden_state_norm: float
    attention_update_norm: float
    ffn_update_norm: float
    total_residual_norm: float
    attention_fraction: float  # ||attn|| / (||attn|| + ||ffn||)
    cosine_similarity_attn_ffn: float
    top_logits: Optional[List[Tuple[str, float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "hidden_state_norm": float(self.hidden_state_norm),
            "attention_update_norm": float(self.attention_update_norm),
            "ffn_update_norm": float(self.ffn_update_norm),
            "total_residual_norm": float(self.total_residual_norm),
            "attention_fraction": float(self.attention_fraction),
            "cosine_similarity_attn_ffn": float(self.cosine_similarity_attn_ffn),
            "top_logits": [(t, float(p)) for t, p in self.top_logits] if self.top_logits else None,
        }


@dataclass
class Vector2DDynamicsTrace:
    """2D Sandbox trajectory trace for a sequence of tokens across transformer layers."""
    token: str
    initial_pos: Tuple[float, float]
    layer_states: List[Tuple[float, float]]  # (x_l, y_l)
    attention_deltas: List[Tuple[float, float]]  # (dx_attn, dy_attn)
    ffn_deltas: List[Tuple[float, float]]  # (dx_ffn, dy_ffn)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token": self.token,
            "initial_pos": [float(self.initial_pos[0]), float(self.initial_pos[1])],
            "layer_states": [[float(x), float(y)] for x, y in self.layer_states],
            "attention_deltas": [[float(x), float(y)] for x, y in self.attention_deltas],
            "ffn_deltas": [[float(x), float(y)] for x, y in self.ffn_deltas],
        }


@VISUALIZER_REGISTRY.register("residual_stream_tracer")
class ResidualStreamTracer(BaseVisualizer):
    """
    Tracer and diagnostic suite for the Transformer Residual Stream Highway.
    Monitors layer-wise norm growth, vector update arithmetic, 2D vector dynamics sandbox,
    and Logit Lens / Logit Prisms linear projections.
    """

    def __init__(self, output_dir: str = "visualizations/residual_stream") -> None:
        super().__init__(output_dir)
        self.cmap = create_custom_colormap()
        set_plot_style()

    def trace_residual_stream(
        self,
        hidden_states: List[Union[torch.Tensor, np.ndarray]],
        attn_outputs: List[Union[torch.Tensor, np.ndarray]],
        ffn_outputs: List[Union[torch.Tensor, np.ndarray]],
        unembed_weight: Optional[Union[torch.Tensor, np.ndarray]] = None,
        vocab: Optional[List[str]] = None,
    ) -> List[ResidualLayerTrace]:
        """
        Traces hidden states and component updates across all transformer layers.

        Args:
            hidden_states: List of L+1 hidden state tensors of shape (N, d_model).
            attn_outputs: List of L attention output tensors of shape (N, d_model).
            ffn_outputs: List of L FFN output tensors of shape (N, d_model).
            unembed_weight: Optional unembedding matrix of shape (vocab_size, d_model).
            vocab: Optional list of vocabulary tokens.

        Returns:
            List of ResidualLayerTrace objects for each layer.
        """
        num_layers = len(attn_outputs)
        traces = []

        unembed_np = _to_numpy(unembed_weight) if unembed_weight is not None else None

        for l in range(num_layers):
            h_curr = _to_numpy(hidden_states[l])
            attn = _to_numpy(attn_outputs[l])
            ffn = _to_numpy(ffn_outputs[l])
            h_next = _to_numpy(hidden_states[l + 1])

            norm_h = float(np.mean(np.linalg.norm(h_curr, axis=-1)))
            norm_attn = float(np.mean(np.linalg.norm(attn, axis=-1)))
            norm_ffn = float(np.mean(np.linalg.norm(ffn, axis=-1)))
            norm_total = float(np.mean(np.linalg.norm(h_next, axis=-1)))

            denom = norm_attn + norm_ffn
            attn_frac = float(norm_attn / denom) if denom > 1e-8 else 0.5

            # Cosine similarity between attention and FFN updates
            dot_prod = np.sum(attn * ffn, axis=-1)
            norms_prod = np.linalg.norm(attn, axis=-1) * np.linalg.norm(ffn, axis=-1)
            cos_sim = float(np.mean(dot_prod / np.maximum(norms_prod, 1e-8)))

            # Logit Lens projection if unembedding is provided
            top_logits = None
            if unembed_np is not None and vocab is not None:
                last_token_h = h_next[-1]  # (d_model,)
                logits = np.matmul(unembed_np, last_token_h)
                probs = np.exp(logits - np.max(logits))
                probs = probs / np.sum(probs)
                top_indices = np.argsort(probs)[-5:][::-1]
                top_logits = [(vocab[idx], float(probs[idx])) for idx in top_indices]

            traces.append(
                ResidualLayerTrace(
                    layer_index=l,
                    hidden_state_norm=norm_h,
                    attention_update_norm=norm_attn,
                    ffn_update_norm=norm_ffn,
                    total_residual_norm=norm_total,
                    attention_fraction=attn_frac,
                    cosine_similarity_attn_ffn=cos_sim,
                    top_logits=top_logits,
                )
            )

        return traces

    def compute_direction_consistency_matrix(
        self,
        layer_updates: List[Union[torch.Tensor, np.ndarray]],
    ) -> np.ndarray:
        r"""
        Computes pairwise cosine similarity between layer update vectors \Delta x_i and \Delta x_j.
        High off-diagonal values indicate layers pushing representations in aligned directions.
        """
        num_layers = len(layer_updates)
        flat_updates = []
        for upd in layer_updates:
            u_np = _to_numpy(upd)
            flat = np.mean(u_np, axis=0) if u_np.ndim > 1 else u_np
            flat_updates.append(_safe_normalize(flat[np.newaxis, :], axis=-1)[0])

        mat = np.zeros((num_layers, num_layers), dtype=np.float32)
        for i in range(num_layers):
            for j in range(num_layers):
                mat[i, j] = float(np.dot(flat_updates[i], flat_updates[j]))
        return mat

    def simulate_2d_vector_dynamics(
        self,
        tokens: List[str],
        num_layers: int = 6,
        seed: int = 42,
    ) -> List[Vector2DDynamicsTrace]:
        """
        Simulates / projects 2D vector dynamics sandbox showing token trajectories
        through memorization, circuit formation, and cleanup phases.
        """
        rng = np.random.default_rng(seed)
        traces = []
        num_tokens = len(tokens)

        for i, token in enumerate(tokens):
            angle = (2.0 * math.pi * i) / max(num_tokens, 1)
            init_x = float(math.cos(angle) * 1.5 + rng.normal(0, 0.1))
            init_y = float(math.sin(angle) * 1.5 + rng.normal(0, 0.1))

            states = [(init_x, init_y)]
            attn_deltas = []
            ffn_deltas = []

            curr_x, curr_y = init_x, init_y
            for l in range(num_layers):
                # Phase-dependent dynamics (early: clustering/memorization, mid: circuits, late: separation)
                phase_factor = (l + 1) / float(num_layers)
                dx_attn = float(rng.normal(0, 0.3) - 0.2 * curr_x * phase_factor)
                dy_attn = float(rng.normal(0, 0.3) - 0.2 * curr_y * phase_factor)

                dx_ffn = float(rng.normal(0, 0.2) + 0.1 * curr_y)
                dy_ffn = float(rng.normal(0, 0.2) - 0.1 * curr_x)

                attn_deltas.append((dx_attn, dy_attn))
                ffn_deltas.append((dx_ffn, dy_ffn))

                curr_x += dx_attn + dx_ffn
                curr_y += dy_attn + dy_ffn
                states.append((curr_x, curr_y))

            traces.append(
                Vector2DDynamicsTrace(
                    token=token,
                    initial_pos=(init_x, init_y),
                    layer_states=states,
                    attention_deltas=attn_deltas,
                    ffn_deltas=ffn_deltas,
                )
            )
        return traces

    def compute_logit_prisms_decomposition(
        self,
        component_contributions: Dict[str, Union[torch.Tensor, np.ndarray]],
        unembed_weight: Union[torch.Tensor, np.ndarray],
        target_token_idx: int,
    ) -> Dict[str, float]:
        """
        Treats intermediate activations linearly to project each component
        (Embedding, Attn_0...Attn_L, FFN_0...FFN_L) onto the final target token logit.
        """
        unembed_np = _to_numpy(unembed_weight)
        target_dir = unembed_np[target_token_idx]  # (d_model,)

        contributions = {}
        for name, comp_vec in component_contributions.items():
            vec = _to_numpy(comp_vec)
            if vec.ndim > 1:
                vec = vec[-1]  # last token position
            logit_contrib = float(np.dot(target_dir, vec))
            contributions[name] = logit_contrib

        return contributions

    def generate(
        self,
        hidden_states: List[Union[torch.Tensor, np.ndarray]],
        attn_outputs: List[Union[torch.Tensor, np.ndarray]],
        ffn_outputs: List[Union[torch.Tensor, np.ndarray]],
        save_name: str = "residual_stream_analysis.png",
    ) -> str:
        """Generates multi-panel figure for the residual stream highway."""
        traces = self.trace_residual_stream(hidden_states, attn_outputs, ffn_outputs)
        layers = [t.layer_index for t in traces]
        h_norms = [t.hidden_state_norm for t in traces]
        attn_norms = [t.attention_update_norm for t in traces]
        ffn_norms = [t.ffn_update_norm for t in traces]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Norm Growth Line Chart
        axes[0].plot(layers, h_norms, "o-", color="royalblue", label="Residual State L2 Norm", lw=2)
        axes[0].plot(layers, attn_norms, "s--", color="deepskyblue", label="Attention Update Norm", lw=1.5)
        axes[0].plot(layers, ffn_norms, "^--", color="limegreen", label="FFN Update Norm", lw=1.5)
        axes[0].set_title("Layer-wise Residual Stream Norm Growth")
        axes[0].set_xlabel("Layer Index")
        axes[0].set_ylabel("L2 Norm")
        axes[0].legend()
        axes[0].grid(True, ls="--", alpha=0.5)

        # 2. Update Contribution Stacked Bar Chart
        attn_fracs = [t.attention_fraction * 100.0 for t in traces]
        ffn_fracs = [(1.0 - t.attention_fraction) * 100.0 for t in traces]

        axes[1].bar(layers, attn_fracs, label="Attention %", color="deepskyblue", alpha=0.85)
        axes[1].bar(layers, ffn_fracs, bottom=attn_fracs, label="FFN %", color="limegreen", alpha=0.85)
        axes[1].set_title("Residual Stream Component Dominance (Attn vs FFN)")
        axes[1].set_xlabel("Layer Index")
        axes[1].set_ylabel("Contribution Percentage (%)")
        axes[1].legend()

        plt.tight_layout()
        return self._save_figure(fig, save_name)


# =====================================================================
# 4. Layer Normalization Dynamics
# =====================================================================

@dataclass
class LayerNormStatistics:
    """Statistical summary of hidden states before and after LayerNorm/RMSNorm."""
    layer_index: int
    pre_mean: float
    pre_variance: float
    pre_rms: float
    post_mean: float
    post_variance: float
    post_rms: float
    cosine_similarity_pre_post: float
    gamma_drift_l2: float
    beta_drift_l2: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "pre_mean": float(self.pre_mean),
            "pre_variance": float(self.pre_variance),
            "pre_rms": float(self.pre_rms),
            "post_mean": float(self.post_mean),
            "post_variance": float(self.post_variance),
            "post_rms": float(self.post_rms),
            "cosine_similarity_pre_post": float(self.cosine_similarity_pre_post),
            "gamma_drift_l2": float(self.gamma_drift_l2),
            "beta_drift_l2": float(self.beta_drift_l2),
        }


@VISUALIZER_REGISTRY.register("layer_normalization_dynamics")
class LayerNormDynamics(BaseVisualizer):
    """
    Interactive laboratory for Layer Normalization and RMSNorm dynamics.
    Analyzes mean/variance scaling, gain (γ) / bias (β) parameter drift,
    hypersphere geometric projection, and Pre-LN vs Post-LN stability.
    """

    def __init__(self, output_dir: str = "visualizations/layer_norm") -> None:
        super().__init__(output_dir)
        self.cmap = create_custom_colormap()
        set_plot_style()

    def compute_layer_norm(
        self,
        x: Union[torch.Tensor, np.ndarray],
        gamma: Optional[Union[torch.Tensor, np.ndarray]] = None,
        beta: Optional[Union[torch.Tensor, np.ndarray]] = None,
        eps: float = 1e-5,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        r"""
        Computes standard Layer Normalization:
            \mu = mean(x, axis=-1)
            \sigma^2 = var(x, axis=-1)
            \hat{x} = (x - \mu) / sqrt(\sigma^2 + \epsilon)
            y = \gamma * \hat{x} + \beta
        """
        x_np = _to_numpy(x)
        mean = np.mean(x_np, axis=-1, keepdims=True)
        var = np.var(x_np, axis=-1, keepdims=True)
        normed = (x_np - mean) / np.sqrt(var + eps)

        out = normed
        if gamma is not None:
            g = _to_numpy(gamma)
            out = out * g
        if beta is not None:
            b = _to_numpy(beta)
            out = out + b

        intermediate = {
            "mean": mean,
            "variance": var,
            "rms": np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True)),
            "normalized_hat": normed,
            "output": out,
        }
        return out, intermediate

    def compute_rms_norm(
        self,
        x: Union[torch.Tensor, np.ndarray],
        gamma: Optional[Union[torch.Tensor, np.ndarray]] = None,
        eps: float = 1e-5,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        r"""
        Computes Root Mean Square Normalization (RMSNorm):
            RMS(x) = sqrt(mean(x^2, axis=-1) + \epsilon)
            \hat{x} = x / RMS(x)
            y = \gamma * \hat{x}
        """
        x_np = _to_numpy(x)
        rms = np.sqrt(np.mean(x_np ** 2, axis=-1, keepdims=True) + eps)
        normed = x_np / rms

        out = normed
        if gamma is not None:
            g = _to_numpy(gamma)
            out = out * g

        intermediate = {
            "mean": np.mean(x_np, axis=-1, keepdims=True),
            "variance": np.var(x_np, axis=-1, keepdims=True),
            "rms": rms,
            "normalized_hat": normed,
            "output": out,
        }
        return out, intermediate

    def analyze_layer_statistics(
        self,
        pre_norm_states: List[Union[torch.Tensor, np.ndarray]],
        post_norm_states: List[Union[torch.Tensor, np.ndarray]],
        gammas: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
        betas: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
    ) -> List[LayerNormStatistics]:
        """Computes comprehensive per-layer statistics before and after normalization."""
        num_layers = len(pre_norm_states)
        stats = []

        for l in range(num_layers):
            pre = _to_numpy(pre_norm_states[l])
            post = _to_numpy(post_norm_states[l])

            pre_mean = float(np.mean(pre))
            pre_var = float(np.var(pre))
            pre_rms = float(np.sqrt(np.mean(pre ** 2)))

            post_mean = float(np.mean(post))
            post_var = float(np.var(post))
            post_rms = float(np.sqrt(np.mean(post ** 2)))

            # Cosine similarity between pre and post vectors
            dot = np.sum(pre * post, axis=-1)
            norm_prod = np.linalg.norm(pre, axis=-1) * np.linalg.norm(post, axis=-1)
            cos_sim = float(np.mean(dot / np.maximum(norm_prod, 1e-8)))

            gamma_drift = 0.0
            if gammas is not None and l < len(gammas):
                g = _to_numpy(gammas[l])
                gamma_drift = float(np.linalg.norm(g - 1.0))

            beta_drift = 0.0
            if betas is not None and l < len(betas):
                b = _to_numpy(betas[l])
                beta_drift = float(np.linalg.norm(b))

            stats.append(
                LayerNormStatistics(
                    layer_index=l,
                    pre_mean=pre_mean,
                    pre_variance=pre_var,
                    pre_rms=pre_rms,
                    post_mean=post_mean,
                    post_variance=post_var,
                    post_rms=post_rms,
                    cosine_similarity_pre_post=cos_sim,
                    gamma_drift_l2=gamma_drift,
                    beta_drift_l2=beta_drift,
                )
            )

        return stats

    def compute_hypersphere_projection_geometry(
        self,
        x: Union[torch.Tensor, np.ndarray],
    ) -> Dict[str, Any]:
        r"""
        Analyzes the geometric projection onto the radius \sqrt{d} hypersphere:
        Decomposes x into mean component x_{\parallel} (along uniform 1 vector)
        and variance component x_{\perp} (orthogonal to 1 vector).
        """
        x_np = _to_numpy(x)
        if x_np.ndim == 1:
            x_np = x_np[np.newaxis, :]
        n, d = x_np.shape

        ones = np.ones(d, dtype=np.float32) / math.sqrt(d)
        mean_proj = np.matmul(x_np, ones)[:, np.newaxis] * ones  # x_parallel
        ortho_comp = x_np - mean_proj  # x_perp

        norm_ortho = np.linalg.norm(ortho_comp, axis=-1, keepdims=True)
        hypersphere_coords = ortho_comp / np.maximum(norm_ortho, 1e-8) * math.sqrt(d)

        # 3D PCA projection for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3, random_state=42)
        all_vecs = np.vstack([x_np, hypersphere_coords])
        coords_3d = pca.fit_transform(all_vecs)

        return {
            "d_model": d,
            "target_sphere_radius": float(math.sqrt(d)),
            "pre_norm_3d": coords_3d[:n].tolist(),
            "post_norm_3d": coords_3d[n:].tolist(),
            "mean_projection_magnitudes": np.linalg.norm(mean_proj, axis=-1).tolist(),
            "orthogonal_magnitudes": norm_ortho.squeeze(-1).tolist(),
        }

    def simulate_pre_vs_post_ln_gradients(
        self,
        num_layers: int = 12,
        num_steps: int = 50,
        seed: int = 42,
    ) -> Dict[str, np.ndarray]:
        """
        Simulates theoretical gradient norm stability comparison between Pre-LN and Post-LN architectures.
        Post-LN exhibits vanishing middle gradients and unstable loss spikes,
        whereas Pre-LN maintains uniform gradient scaling.
        """
        rng = np.random.default_rng(seed)
        layers = np.arange(num_layers)

        # Post-LN: U-shaped gradient curve with high variance
        post_ln_grads = np.zeros((num_steps, num_layers), dtype=np.float32)
        # Pre-LN: Stable, flatter gradient distribution
        pre_ln_grads = np.zeros((num_steps, num_layers), dtype=np.float32)

        for step in range(num_steps):
            u_shape = 0.5 * (layers - (num_layers - 1) / 2.0) ** 2 / (num_layers) + 0.1
            spike = 3.0 if step in (12, 33) else 1.0
            post_ln_grads[step] = (u_shape * spike + rng.normal(0, 0.05, size=num_layers)).clip(min=0.01)

            pre_flat = 0.8 + 0.05 * np.sin(step * 0.2) + rng.normal(0, 0.02, size=num_layers)
            pre_ln_grads[step] = pre_flat.clip(min=0.1)

        return {
            "post_ln_gradients": post_ln_grads,
            "pre_ln_gradients": pre_ln_grads,
            "layers": layers,
        }

    def generate(
        self,
        pre_norm_states: List[Union[torch.Tensor, np.ndarray]],
        post_norm_states: List[Union[torch.Tensor, np.ndarray]],
        save_name: str = "layer_norm_dynamics.png",
    ) -> str:
        """Generates summary visualization for LayerNorm statistics."""
        stats = self.analyze_layer_statistics(pre_norm_states, post_norm_states)
        layers = [s.layer_index for s in stats]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Variance Scaling Pre vs Post
        axes[0].plot(layers, [s.pre_variance for s in stats], "o-", color="crimson", label="Pre-LN Variance", lw=2)
        axes[0].plot(layers, [s.post_variance for s in stats], "s-", color="mediumseagreen", label="Post-LN Variance (≈1.0)", lw=2)
        axes[0].set_title("Layer-wise Activation Variance Stabilization")
        axes[0].set_xlabel("Layer Index")
        axes[0].set_ylabel("Variance")
        axes[0].legend()
        axes[0].grid(True, ls="--", alpha=0.5)

        # 2. Pre/Post Cosine Similarity
        axes[1].plot(layers, [s.cosine_similarity_pre_post for s in stats], "^-", color="darkorange", lw=2)
        axes[1].set_title("Pre-LN vs Post-LN Directional Preservation cos(x, x̂)")
        axes[1].set_xlabel("Layer Index")
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_ylim([0.0, 1.05])
        axes[1].grid(True, ls="--", alpha=0.5)

        plt.tight_layout()
        return self._save_figure(fig, save_name)
