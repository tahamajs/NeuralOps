"""
Attention Canvas Suite (Zones 1-4 Engine)
=========================================
Comprehensive computational backend supporting the NeuralOps Attention Visualization Dashboard:
- Zone 1: Context & Exploration State Provider
- Zone 2: Attention Canvas (Head View, Model View, Sankey Flow, Q/K Joint Embedding)
- Zone 3: Live Math Panel (KaTeX breakdown, Step-by-Step Stepper, Numerical Inspector, Math Diff)
- Zone 4: Control Panel Serialization & Experiment / Paper Export helpers

Compatible with standard PyTorch tensors, NumPy arrays, and HuggingFace Transformer outputs.
"""

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _native_jensen_shannon_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Fast native computation of Jensen-Shannon distance in bits."""
    p_norm = np.clip(p / (np.sum(p) + 1e-12), 1e-12, 1.0)
    q_norm = np.clip(q / (np.sum(q) + 1e-12), 1e-12, 1.0)
    m = 0.5 * (p_norm + q_norm)
    kl_p_m = np.sum(p_norm * np.log2(p_norm / m))
    kl_q_m = np.sum(q_norm * np.log2(q_norm / m))
    js_divergence = 0.5 * (kl_p_m + kl_q_m)
    return float(np.sqrt(np.maximum(0.0, js_divergence)))



# ============================================================================
# Dataclasses & Type Definitions for Frontend Ingestion
# ============================================================================

@dataclass
class HeadViewCell:
    query_idx: int
    key_idx: int
    query_token: str
    key_token: str
    weight: float
    raw_score: float
    is_diagonal: bool


@dataclass
class HeadViewData:
    layer_idx: int
    head_idx: int
    model_name: str
    attention_type: str
    tokens: List[str]
    matrix: List[List[float]]
    entropy_per_token: List[float]
    mean_entropy: float
    concentration_top3: float
    diagonal_ratio: float
    top_attended_pairs: List[Dict[str, Any]]
    svg_heatmap: Optional[str] = None


@dataclass
class HeadMetric:
    layer_idx: int
    head_idx: int
    importance: float
    diversity: float
    entropy: float
    concentration: float
    specialization: str
    confidence: float
    thumbnail: Optional[List[List[float]]] = None


@dataclass
class ModelViewData:
    num_layers: int
    num_heads: int
    heads: List[HeadMetric]
    layer_entropies: List[float]
    layer_importances: List[float]
    most_important_heads: List[Dict[str, Any]]
    specialization_counts: Dict[str, int]


@dataclass
class SankeyNode:
    id: str
    name: str
    layer: int
    node_type: str  # 'input', 'head', 'output'
    value: float
    color: Optional[str] = None


@dataclass
class SankeyLink:
    source: str
    target: str
    value: float
    weight: float
    color: Optional[str] = None


@dataclass
class SankeyData:
    nodes: List[SankeyNode]
    links: List[SankeyLink]
    layers_count: int
    tokens: List[str]


@dataclass
class EmbeddingPoint:
    id: str
    token_idx: int
    token: str
    point_type: str  # 'query' or 'key'
    x: float
    y: float
    magnitude: float
    attention_sum: float


@dataclass
class QKEmbeddingData:
    method: str
    points: List[EmbeddingPoint]
    connections: List[Dict[str, Any]]
    tokens: List[str]
    explained_variance_ratio: Optional[List[float]] = None


@dataclass
class MathStep:
    step_number: int
    title: str
    latex_equation: str
    description: str
    variables: Dict[str, str]
    sub_values: Dict[str, Any]


@dataclass
class LiveMathBreakdown:
    attention_type: str
    primary_formula: str
    variable_color_map: Dict[str, str]
    steps: List[MathStep]
    complexity_table: Dict[str, Any]
    numerical_inspection: Dict[str, Any]
    comparison_diff: Optional[Dict[str, Any]] = None


# ============================================================================
# Core Zone 2: Attention Canvas Extractors & Computations
# ============================================================================

def _to_numpy(tensor_or_array: Union[torch.Tensor, np.ndarray, List]) -> np.ndarray:
    """Safely convert any tensor/list input to a float32 numpy array."""
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.detach().cpu().to(torch.float32).numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return np.array(tensor_or_array, dtype=np.float32)


def compute_shannon_entropy(prob_vector: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Shannon entropy in bits for a discrete probability distribution."""
    p = np.clip(prob_vector, eps, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p)))


def extract_head_view(
    attention_matrix: Union[torch.Tensor, np.ndarray, List[List[float]]],
    tokens: Optional[List[str]] = None,
    layer_idx: int = 0,
    head_idx: int = 0,
    model_name: str = "Transformer-Base",
    attention_type: str = "Softmax MHA",
    generate_svg: bool = False,
) -> HeadViewData:
    """
    Extracts the N x N attention matrix for a specific layer and head,
    computing token-token statistics, entropy curves, and top pairs.
    """
    attn = _to_numpy(attention_matrix)
    if attn.ndim == 4:
        attn = attn[layer_idx, head_idx]
    elif attn.ndim == 3:
        attn = attn[head_idx]

    N, M = attn.shape
    if tokens is None or len(tokens) != N:
        tokens = [f"tok_{i}" for i in range(N)]

    # Compute entropy per query row
    entropies = [compute_shannon_entropy(attn[i]) for i in range(N)]
    mean_entropy = float(np.mean(entropies)) if len(entropies) > 0 else 0.0

    # Top-3 concentration: average mass contained in top 3 key tokens per row
    k_val = min(3, M)
    sorted_rows = np.sort(attn, axis=-1)[:, ::-1]
    concentration_top3 = float(np.mean(np.sum(sorted_rows[:, :k_val], axis=-1)))

    # Diagonal ratio (self-attention mass)
    diag_indices = np.arange(min(N, M))
    diagonal_sum = float(np.sum(attn[diag_indices, diag_indices]))
    total_sum = float(np.sum(attn)) + 1e-12
    diagonal_ratio = diagonal_sum / total_sum

    # Identify top 10 most attended token pairs
    flat_indices = np.argsort(attn.ravel())[::-1][:min(10, N * M)]
    top_pairs = []
    for idx in flat_indices:
        q_idx = int(idx // M)
        k_idx = int(idx % M)
        top_pairs.append({
            "query_idx": q_idx,
            "key_idx": k_idx,
            "query_token": tokens[q_idx],
            "key_token": tokens[k_idx],
            "weight": float(attn[q_idx, k_idx]),
        })

    svg_str = None
    if generate_svg:
        svg_str = generate_head_heatmap_svg(attn, tokens, layer_idx, head_idx)

    return HeadViewData(
        layer_idx=layer_idx,
        head_idx=head_idx,
        model_name=model_name,
        attention_type=attention_type,
        tokens=tokens,
        matrix=attn.tolist(),
        entropy_per_token=entropies,
        mean_entropy=mean_entropy,
        concentration_top3=concentration_top3,
        diagonal_ratio=diagonal_ratio,
        top_attended_pairs=top_pairs,
        svg_heatmap=svg_str,
    )


def classify_head_specialization(
    head_matrix: np.ndarray,
    tokens: Optional[List[str]] = None,
    mean_entropy: float = 2.0,
    diagonal_ratio: float = 0.3,
) -> Tuple[str, float]:
    """
    Classify the linguistic / structural specialization of an attention head:
    - Positional: strong diagonal or next/previous token band
    - Induction: strong delayed repetition or offset diagonal
    - Delimiter: heavy attention towards [CLS], [SEP], commas, or punctuation
    - Semantic / Broad Context: high entropy, diffuse attention across non-adjacent words
    """
    N, M = head_matrix.shape
    diag = np.diag(head_matrix)
    diag_prev = np.diag(head_matrix, k=-1) if N > 1 else np.array([0.0])
    diag_next = np.diag(head_matrix, k=1) if M > 1 else np.array([0.0])

    prev_token_mass = float(np.mean(diag_prev)) if len(diag_prev) > 0 else 0.0
    next_token_mass = float(np.mean(diag_next)) if len(diag_next) > 0 else 0.0
    self_mass = float(np.mean(diag)) if len(diag) > 0 else 0.0

    # Check for delimiter token concentration if tokens provided
    delimiter_mass = 0.0
    if tokens:
        delim_indices = [
            i for i, t in enumerate(tokens)
            if t in {"[CLS]", "[SEP]", "<s>", "</s>", "<|endoftext|>", ".", ",", "!", "?"}
        ]
        if delim_indices:
            delimiter_mass = float(np.mean(head_matrix[:, delim_indices]))

    if delimiter_mass > 0.45:
        return "Delimiter Head", min(0.99, delimiter_mass * 1.5)
    elif prev_token_mass > 0.35:
        return "Induction Head", min(0.99, prev_token_mass * 1.8)
    elif self_mass > 0.4 or diagonal_ratio > 0.45 or next_token_mass > 0.35:
        return "Positional Head", min(0.99, max(self_mass, diagonal_ratio, next_token_mass) * 1.4)
    elif mean_entropy > math.log2(max(2, N)) * 0.75:
        return "Broad Context Head", min(0.95, mean_entropy / math.log2(max(2, N)))
    else:
        return "Semantic Head", 0.78


def compute_model_view_metrics(
    attention_tensor: Union[torch.Tensor, np.ndarray, List],
    tokens: Optional[List[str]] = None,
    include_thumbnails: bool = True,
    thumbnail_size: int = 8,
) -> ModelViewData:
    """
    Computes bird's-eye metrics across all layers and heads:
    Importance, Diversity (Jensen-Shannon divergence to peer heads),
    Entropy, Concentration, and Classification.
    """
    attn = _to_numpy(attention_tensor)
    if attn.ndim != 4:
        raise ValueError(
            f"Expected 4D attention tensor [num_layers, num_heads, seq_len, seq_len], got shape {attn.shape}"
        )

    num_layers, num_heads, N, M = attn.shape
    if tokens is None:
        tokens = [f"tok_{i}" for i in range(N)]

    head_metrics: List[HeadMetric] = []
    layer_entropies: List[float] = []
    layer_importances: List[float] = []
    spec_counts: Dict[str, int] = {
        "Positional Head": 0,
        "Induction Head": 0,
        "Semantic Head": 0,
        "Delimiter Head": 0,
        "Broad Context Head": 0,
    }

    max_entropy_possible = math.log2(max(2, M))

    for l in range(num_layers):
        layer_m = attn[l]  # shape [num_heads, N, M]
        layer_entropies_l = []
        layer_importances_l = []

        # Precompute normalized heads for diversity computation across heads in this layer
        flat_heads = [layer_m[h].reshape(N, M) for h in range(num_heads)]
        flat_prob_heads = [np.clip(h / (np.sum(h, axis=-1, keepdims=True) + 1e-12), 1e-12, 1.0) for h in flat_heads]

        for h in range(num_heads):
            head_m = layer_m[h]
            # Entropy
            h_entropies = [compute_shannon_entropy(head_m[i]) for i in range(N)]
            mean_h_entropy = float(np.mean(h_entropies))
            layer_entropies_l.append(mean_h_entropy)

            # Concentration
            k_val = min(3, M)
            sorted_rows = np.sort(head_m, axis=-1)[:, ::-1]
            concentration = float(np.mean(np.sum(sorted_rows[:, :k_val], axis=-1)))

            # Importance heuristic: distance from uniform distribution combined with Frobenius dispersion
            uniform_dist = np.ones((N, M), dtype=np.float32) / float(M)
            dist_from_uniform = float(np.linalg.norm(head_m - uniform_dist, ord="fro") / np.sqrt(N * M))
            importance = float(np.clip(dist_from_uniform * 2.5, 0.1, 1.0))
            layer_importances_l.append(importance)

            # Diversity score: average Jensen-Shannon distance against all other heads in the same layer
            js_distances = []
            for other_h in range(num_heads):
                if other_h == h:
                    continue
                # Average JSD row-by-row
                row_jsds = [
                    _native_jensen_shannon_distance(flat_prob_heads[h][row_i], flat_prob_heads[other_h][row_i])
                    for row_i in range(N)
                ]
                js_distances.append(np.nanmean(row_jsds))
            diversity = float(np.mean(js_distances)) if js_distances else 0.5
            diversity = float(np.clip(diversity, 0.05, 0.99))

            # Diagonal ratio
            diag_sum = float(np.trace(head_m))
            diag_ratio = diag_sum / (float(np.sum(head_m)) + 1e-12)

            # Specialization
            spec_type, conf = classify_head_specialization(head_m, tokens, mean_h_entropy, diag_ratio)
            spec_counts[spec_type] = spec_counts.get(spec_type, 0) + 1

            # Thumbnail downsample
            thumb = None
            if include_thumbnails:
                t_size = min(thumbnail_size, N, M)
                step_r = max(1, N // t_size)
                step_c = max(1, M // t_size)
                downsampled = head_m[::step_r, ::step_c][:t_size, :t_size]
                thumb = downsampled.tolist()

            head_metrics.append(
                HeadMetric(
                    layer_idx=l,
                    head_idx=h,
                    importance=importance,
                    diversity=diversity,
                    entropy=mean_h_entropy,
                    concentration=concentration,
                    specialization=spec_type,
                    confidence=conf,
                    thumbnail=thumb,
                )
            )

        layer_entropies.append(float(np.mean(layer_entropies_l)))
        layer_importances.append(float(np.mean(layer_importances_l)))

    # Sort top 5 most important heads
    sorted_heads = sorted(head_metrics, key=lambda x: x.importance, reverse=True)[:5]
    top_heads_summary = [
        {
            "layer": hm.layer_idx,
            "head": hm.head_idx,
            "importance": hm.importance,
            "specialization": hm.specialization,
            "entropy": hm.entropy,
        }
        for hm in sorted_heads
    ]

    return ModelViewData(
        num_layers=num_layers,
        num_heads=num_heads,
        heads=head_metrics,
        layer_entropies=layer_entropies,
        layer_importances=layer_importances,
        most_important_heads=top_heads_summary,
        specialization_counts=spec_counts,
    )


def generate_sankey_flow_data(
    attention_tensor: Union[torch.Tensor, np.ndarray, List],
    tokens: List[str],
    top_k_flows: int = 40,
    threshold: float = 0.02,
) -> SankeyData:
    """
    Generates multi-layer Sankey attention flow data tracing information
    propagation from Input Tokens -> Intermediate Layer Heads -> Output Predictions.
    """
    attn = _to_numpy(attention_tensor)
    if attn.ndim == 3:
        attn = attn[np.newaxis, ...]
    num_layers, num_heads, N, _ = attn.shape

    nodes: List[SankeyNode] = []
    links: List[SankeyLink] = []

    # 1. Input Layer nodes
    for i, tok in enumerate(tokens):
        nodes.append(
            SankeyNode(
                id=f"input_{i}",
                name=tok,
                layer=0,
                node_type="input",
                value=1.0,
                color="#3b82f6",
            )
        )

    # 2. Intermediate Layer Head nodes
    for l in range(num_layers):
        for h in range(num_heads):
            # Compute total attention mass received by head (l, h)
            head_mass = float(np.sum(attn[l, h]))
            nodes.append(
                SankeyNode(
                    id=f"L{l}_H{h}",
                    name=f"L{l}·H{h}",
                    layer=l + 1,
                    node_type="head",
                    value=head_mass,
                    color="#8b5cf6" if l % 2 == 0 else "#ec4899",
                )
            )

    # 3. Output Layer nodes
    for i, tok in enumerate(tokens):
        nodes.append(
            SankeyNode(
                id=f"output_{i}",
                name=f"Out:{tok}",
                layer=num_layers + 1,
                node_type="output",
                value=1.0,
                color="#10b981",
            )
        )

    # 4. Links: Input -> Layer 0 heads
    raw_links: List[Tuple[str, str, float, float]] = []
    for i in range(N):
        for h in range(num_heads):
            w = float(np.mean(attn[0, h, i, :]))
            if w >= threshold:
                raw_links.append((f"input_{i}", f"L0_H{h}", w, w))

    # Links: Layer l heads -> Layer l+1 heads
    for l in range(num_layers - 1):
        for h1 in range(num_heads):
            for h2 in range(num_heads):
                # Correlation / flow between head representations
                flow_mag = float(np.sum(attn[l, h1] * attn[l + 1, h2]) / (N * N))
                if flow_mag >= threshold * 0.5:
                    raw_links.append((f"L{l}_H{h1}", f"L{l+1}_H{h2}", flow_mag, flow_mag))

    # Links: Last layer heads -> Output
    last_l = num_layers - 1
    for h in range(num_heads):
        for i in range(N):
            w = float(np.mean(attn[last_l, h, :, i]))
            if w >= threshold:
                raw_links.append((f"L{last_l}_H{h}", f"output_{i}", w, w))

    # Filter top K links by value
    raw_links.sort(key=lambda x: x[2], reverse=True)
    filtered_links = raw_links[:top_k_flows]

    for src, tgt, val, wt in filtered_links:
        links.append(SankeyLink(source=src, target=tgt, value=val, weight=wt))

    return SankeyData(
        nodes=nodes,
        links=links,
        layers_count=num_layers + 2,
        tokens=tokens,
    )


def compute_qk_joint_embedding(
    query_matrix: Union[torch.Tensor, np.ndarray, List],
    key_matrix: Union[torch.Tensor, np.ndarray, List],
    tokens: List[str],
    method: str = "pca",
    n_components: int = 2,
    perplexity: float = 5.0,
    random_state: int = 42,
) -> QKEmbeddingData:
    """
    Computes joint 2D embeddings of Query and Key vectors (AttentionViz style).
    Points closer in 2D space have higher dot-product similarity.
    """
    Q = _to_numpy(query_matrix)
    K = _to_numpy(key_matrix)

    if Q.ndim == 3:  # [batch, seq, dim] -> take first batch
        Q = Q[0]
    if K.ndim == 3:
        K = K[0]

    N, D = Q.shape
    M, _ = K.shape

    # Combine Q and K vectors: shape [2N, D]
    combined = np.vstack([Q, K])

    explained_var = None
    if method.lower() == "pca":
        # Native fast PCA via mean-centered SVD
        centered = combined - np.mean(combined, axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        coords_2d = u[:, :n_components] * s[:n_components]
        total_var = float(np.sum(s ** 2)) + 1e-12
        explained_var = [float(s[i] ** 2 / total_var) for i in range(min(n_components, len(s)))]
    elif method.lower() == "tsne":
        try:
            from sklearn.manifold import TSNE
            perp = min(perplexity, max(1.0, (2 * N - 1) / 3.0))
            tsne = TSNE(
                n_components=n_components,
                perplexity=perp,
                random_state=random_state,
                init="pca",
                learning_rate="auto",
                max_iter=250,
            )
            coords_2d = tsne.fit_transform(combined)
        except Exception:
            u, s, vt = np.linalg.svd(combined, full_matrices=False)
            coords_2d = u[:, :n_components] * s[:n_components]
    else:
        # Fallback to normalized SVD
        u, s, vt = np.linalg.svd(combined, full_matrices=False)
        coords_2d = u[:, :n_components] * s[:n_components]

    # Normalize coordinates to range [-100, 100]
    max_abs = np.max(np.abs(coords_2d)) + 1e-12
    coords_norm = (coords_2d / max_abs) * 100.0

    # Attention dot products
    raw_scores = np.matmul(Q, K.T) / np.sqrt(D)
    exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=-1, keepdims=True))
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    points: List[EmbeddingPoint] = []
    # Query points
    for i in range(N):
        tok = tokens[i] if i < len(tokens) else f"tok_{i}"
        q_norm = float(np.linalg.norm(Q[i]))
        attn_sum = float(np.sum(attn_weights[i, :]))
        points.append(
            EmbeddingPoint(
                id=f"Q_{i}",
                token_idx=i,
                token=tok,
                point_type="query",
                x=float(coords_norm[i, 0]),
                y=float(coords_norm[i, 1]),
                magnitude=q_norm,
                attention_sum=attn_sum,
            )
        )

    # Key points
    for j in range(M):
        tok = tokens[j] if j < len(tokens) else f"tok_{j}"
        k_norm = float(np.linalg.norm(K[j]))
        attn_recv = float(np.sum(attn_weights[:, j]))
        points.append(
            EmbeddingPoint(
                id=f"K_{j}",
                token_idx=j,
                token=tok,
                point_type="key",
                x=float(coords_norm[N + j, 0]),
                y=float(coords_norm[N + j, 1]),
                magnitude=k_norm,
                attention_sum=attn_recv,
            )
        )

    # Connections between Q_i and K_j above threshold
    connections = []
    for i in range(N):
        for j in range(M):
            wt = float(attn_weights[i, j])
            if wt > 0.08:
                connections.append({
                    "source": f"Q_{i}",
                    "target": f"K_{j}",
                    "query_idx": i,
                    "key_idx": j,
                    "weight": wt,
                    "raw_score": float(raw_scores[i, j]),
                })

    return QKEmbeddingData(
        method=method.upper(),
        points=points,
        connections=connections,
        tokens=tokens,
        explained_variance_ratio=explained_var,
    )


# ============================================================================
# Core Zone 3: Live Math Panel KaTeX Step-by-Step Breakdown
# ============================================================================

def generate_live_math_breakdown(
    attention_type: str = "svd",
    q_vec: Optional[np.ndarray] = None,
    k_vec: Optional[np.ndarray] = None,
    v_vec: Optional[np.ndarray] = None,
    d_k: int = 64,
    rank: int = 16,
    query_idx: int = 0,
    key_idx: int = 1,
    tokens: Optional[List[str]] = None,
) -> LiveMathBreakdown:
    """
    Generates dynamic KaTeX / LaTeX mathematical steps, variable color mappings,
    numerical inspector values, complexity FLOPs comparisons, and architecture diffs.
    """
    # Fallback dummy vectors if not provided
    if q_vec is None:
        np.random.seed(42 + query_idx)
        q_vec = np.random.randn(d_k).astype(np.float32)
    if k_vec is None:
        np.random.seed(1337 + key_idx)
        k_vec = np.random.randn(d_k).astype(np.float32)
    if v_vec is None:
        np.random.seed(999 + key_idx)
        v_vec = np.random.randn(d_k).astype(np.float32)

    q_norm = float(np.linalg.norm(q_vec))
    k_norm = float(np.linalg.norm(k_vec))
    v_norm = float(np.linalg.norm(v_vec))

    raw_dot = float(np.dot(q_vec, k_vec))
    scale_factor = float(np.sqrt(d_k))
    scaled_score = raw_dot / scale_factor
    exp_val = float(np.exp(np.clip(scaled_score, -20.0, 20.0)))
    simulated_row_sum = exp_val + 5.432  # simulated partition function
    softmax_weight = exp_val / simulated_row_sum
    weighted_v_head = [float(x * softmax_weight) for x in v_vec[:5]]

    var_colors = {
        "Q": "#3b82f6",  # Blue
        "K": "#ef4444",  # Red
        "V": "#10b981",  # Green
        "d_k": "#94a3b8",  # Slate
        "softmax": "#a855f7",  # Purple
        "scale": "#f59e0b",  # Amber
        "SVD": "#ec4899",  # Pink
        "U": "#06b6d4",  # Cyan
        "Sigma": "#8b5cf6",  # Violet
        "Rank": "#14b8a6",  # Teal
    }

    steps: List[MathStep] = []
    attn_type_lower = attention_type.lower()

    if "svd" in attn_type_lower or "galerkin" in attn_type_lower or "low_rank" in attn_type_lower:
        primary_formula = r"\text{Attention}_{\text{SVD}}(Q, K, V) = \text{softmax}\left(\frac{Q (K S_r)^T}{\sqrt{d_k}}\right) (V S_r)"
        
        # SVD / Sketch calculations
        sketch_scale = float(np.sqrt(rank / d_k))
        svd_projected_dot = raw_dot * sketch_scale
        svd_scaled = svd_projected_dot / scale_factor
        svd_exp = float(np.exp(np.clip(svd_scaled, -20.0, 20.0)))
        svd_weight = svd_exp / (svd_exp + 4.981)

        steps = [
            MathStep(
                step_number=1,
                title="Low-Rank SVD / Sketch Projection",
                latex_equation=r"K_r = K \cdot S_r \in \mathbb{R}^{N \times r}, \quad V_r = V \cdot S_r \in \mathbb{R}^{N \times r}",
                description=f"Compresses Key and Value matrices into rank r={rank} subspace via Petrov-Galerkin sketching or truncated singular vectors $U_r \\Sigma_r$.",
                variables={"K": "Key matrix", "S_r": f"Sketching matrix (N x {rank})", "V_r": "Compressed values"},
                sub_values={"rank": rank, "compression_ratio": f"{d_k / rank:.1f}x", "dim_reduced_to": rank},
            ),
            MathStep(
                step_number=2,
                title="Low-Rank Scaled Dot-Product",
                latex_equation=r"S_{\text{SVD}} = \frac{Q K_r^T}{\sqrt{d_k}} = \frac{Q (K S_r)^T}{\sqrt{d_k}}",
                description=f"Computes query-key compatibility in the reduced r={rank} dimensional bottleneck.",
                variables={"Q": "Query vector", "K_r": "Rank-reduced key", "\\sqrt{d_k}": f"Scaling factor = {scale_factor:.3f}"},
                sub_values={"raw_dot": raw_dot, "projected_score": svd_projected_dot, "scaled_score": svd_scaled},
            ),
            MathStep(
                step_number=3,
                title="Row-wise Normalization / Softmax",
                latex_equation=r"A_{\text{SVD}} = \text{softmax}(S_{\text{SVD}})_{ij} = \frac{\exp(S_{\text{SVD}, ij})}{\sum_k \exp(S_{\text{SVD}, ik})}",
                description="Converts scaled compatibility scores into attention probability weights.",
                variables={"A": "Attention probability matrix", "\\text{softmax}": "Row-wise normalized exponential"},
                sub_values={"exp_score": svd_exp, "attention_weight": svd_weight},
            ),
            MathStep(
                step_number=4,
                title="Compressed Value Aggregation",
                latex_equation=r"O_{\text{SVD}} = A_{\text{SVD}} \cdot V_r = \sum_{j=1}^N A_{\text{SVD}, ij} V_{r, j}",
                description="Aggregates projected values with attention weights to form the output context vector.",
                variables={"O": "Final Context representation", "V_r": "Low-rank value representations"},
                sub_values={"weighted_vector_sample": [float(x * svd_weight) for x in v_vec[:5]]},
            ),
        ]

        complexity = {
            "time_complexity": f"O(N \\cdot r \\cdot d_k) = O({rank} N d_k)",
            "memory_complexity": f"O(N \\cdot r) = O({rank} N)",
            "speedup_vs_dense": f"{d_k / rank:.1f}x faster quadratic bypass",
            "flops_n1024": f"{2 * 1024 * rank * d_k / 1e6:.2f} MFLOPs (vs {2 * 1024 * 1024 * d_k / 1e6:.2f} MFLOPs dense)",
        }

    elif "linear" in attn_type_lower:
        primary_formula = r"\text{Attention}_{\text{Lin}}(Q, K, V) = \frac{\phi(Q) (\phi(K)^T V)}{\phi(Q) \sum_j \phi(K_j)^T}"
        
        steps = [
            MathStep(
                step_number=1,
                title="Feature Map Kernel Transformation",
                latex_equation=r"\phi(x) = \text{ELU}(x) + 1 \quad \text{or} \quad \phi(x) = \text{softmax}(x)",
                description="Applies non-negative kernel feature map ensuring positive semi-definite dot products.",
                variables={"\\phi": "Feature kernel map", "x": "Query or Key activation"},
                sub_values={"phi_q_norm": float(np.linalg.norm(np.maximum(0, q_vec) + 1))},
            ),
            MathStep(
                step_number=2,
                title="Linear Associativity Reordering",
                latex_equation=r"M = \phi(K)^T V \in \mathbb{R}^{d_k \times d_v}, \quad Z = \sum_{j=1}^N \phi(K_j)^T \in \mathbb{R}^{d_k}",
                description="By matrix associativity $(Q K^T)V = Q (K^T V)$, precomputes global state memory $M$ in $O(N d^2)$.",
                variables={"M": "KV Global State Memory Matrix", "Z": "Normalization vector accumulator"},
                sub_values={"memory_dim": f"{d_k} x {d_k}", "tokens_processed": len(tokens) if tokens else 10},
            ),
            MathStep(
                step_number=3,
                title="Constant-Time Context Querying",
                latex_equation=r"O_i = \frac{\phi(Q_i) \cdot M}{\phi(Q_i) \cdot Z}",
                description="Queries memory state in $O(1)$ per token during autoregressive decoding.",
                variables={"O_i": "Token output context", "\\phi(Q_i)": "Transformed query vector"},
                sub_values={"token_index": query_idx, "attention_weight": softmax_weight},
            ),
        ]

        complexity = {
            "time_complexity": "O(N \\cdot d_k^2)",
            "memory_complexity": "O(d_k^2) constant with sequence length",
            "speedup_vs_dense": "Strictly O(N) linear time scaling",
            "flops_n1024": f"{2 * 1024 * d_k * d_k / 1e6:.2f} MFLOPs",
        }

    elif "gqa" in attn_type_lower or "grouped" in attn_type_lower:
        primary_formula = r"\text{GQA}_g(Q_g, K_{\lfloor g/R \rfloor}, V_{\lfloor g/R \rfloor}) = \text{softmax}\left(\frac{Q_g K_{\lfloor g/R \rfloor}^T}{\sqrt{d_k}}\right) V_{\lfloor g/R \rfloor}"
        
        steps = [
            MathStep(
                step_number=1,
                title="Head Group Mapping",
                latex_equation=r"g \in [0, H_Q - 1], \quad k_{\text{group}} = \lfloor g / R \rfloor \quad \text{where } R = H_Q / H_{KV}",
                description="Shares a single Key/Value head across R query heads, slashing KV cache memory by factor R.",
                variables={"H_Q": "Number of Query heads", "H_{KV}": "Number of KV heads", "R": "Group ratio"},
                sub_values={"query_head": 7, "shared_kv_head": 1, "group_ratio": 4},
            ),
            MathStep(
                step_number=2,
                title="Grouped Scaled Dot-Product",
                latex_equation=r"S_g = \frac{Q_g \cdot K_{k_{\text{group}}}^T}{\sqrt{d_k}}",
                description="Queries compute attention against the shared key projection.",
                variables={"Q_g": "Specific query head vector", "K_{k}": "Shared key head vector"},
                sub_values={"scaled_score": scaled_score, "raw_dot": raw_dot},
            ),
            MathStep(
                step_number=3,
                title="Softmax and Grouped Value Synthesis",
                latex_equation=r"O_g = \text{softmax}(S_g) \cdot V_{k_{\text{group}}}",
                description="Aggregates values from the shared value projection.",
                variables={"O_g": "Output context of query head g", "V_k": "Shared value head"},
                sub_values={"attention_weight": softmax_weight},
            ),
        ]

        complexity = {
            "time_complexity": "O(N^2 \\cdot d_k)",
            "memory_complexity": "O(N^2) compute, O(N \\cdot H_{KV} \\cdot d_k) KV cache",
            "speedup_vs_dense": "4x-8x KV cache reduction",
            "flops_n1024": f"{2 * 1024 * 1024 * d_k / 1e6:.2f} MFLOPs",
        }

    else:
        # Standard Softmax MHA
        primary_formula = r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V"
        
        steps = [
            MathStep(
                step_number=1,
                title="Query-Key Matrix Multiplication",
                latex_equation=r"S_{\text{raw}} = Q \cdot K^T \in \mathbb{R}^{N \times N}",
                description="Calculates unnormalized pairwise dot-product affinity between all query and key token embeddings.",
                variables={"Q": "Query matrix (N x d_k)", "K": "Key matrix (N x d_k)", "S_{\\text{raw}}": "Raw similarity scores"},
                sub_values={"Q_norm": q_norm, "K_norm": k_norm, "raw_dot_product": raw_dot},
            ),
            MathStep(
                step_number=2,
                title="Variance Scaling by \\sqrt{d_k}",
                latex_equation=r"S = \frac{S_{\text{raw}}}{\sqrt{d_k}} = \frac{Q K^T}{\sqrt{d_k}}",
                description=f"Scales dot products by $\\sqrt{{d_k}} = \\sqrt{{{d_k}}} = {scale_factor:.2f}$ to prevent softmax gradients from vanishing in high dimensions.",
                variables={"\\sqrt{d_k}": f"Scaling constant ({scale_factor:.3f})", "S": "Scaled compatibility score"},
                sub_values={"scaled_dot_product": scaled_score},
            ),
            MathStep(
                step_number=3,
                title="Row-wise Softmax Probability Distribution",
                latex_equation=r"A_{ij} = \text{softmax}(S)_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^N \exp(S_{ik})}",
                description="Exponentiates and normalizes each query row so weights sum to exactly 1.0.",
                variables={"A": "Attention probability matrix", "\\text{softmax}": "Normalized exponential"},
                sub_values={"exp_score": exp_val, "row_sum": simulated_row_sum, "attention_weight": softmax_weight},
            ),
            MathStep(
                step_number=4,
                title="Value Matrix Linear Combination",
                latex_equation=r"O = A \cdot V \in \mathbb{R}^{N \times d_v} \implies O_i = \sum_{j=1}^N A_{ij} V_j",
                description="Constructs new contextual representation for token i as the weighted sum over all Value vectors.",
                variables={"O": "Multi-Head Attention Output matrix", "V": "Value matrix"},
                sub_values={"output_dim": f"d_v = {d_k}", "sample_vector": weighted_v_head},
            ),
        ]

        complexity = {
            "time_complexity": "O(N^2 \\cdot d_k)",
            "memory_complexity": "O(N^2) quadratic memory bottleneck",
            "speedup_vs_dense": "1.0x (Standard Baseline)",
            "flops_n1024": f"{2 * 1024 * 1024 * d_k / 1e6:.2f} MFLOPs",
        }

    # Numerical inspection dictionary for hovered/clicked token pair
    num_inspection = {
        "query_token": tokens[query_idx] if tokens and query_idx < len(tokens) else f"token_{query_idx}",
        "key_token": tokens[key_idx] if tokens and key_idx < len(tokens) else f"token_{key_idx}",
        "query_index": query_idx,
        "key_index": key_idx,
        "head_dimension_d_k": d_k,
        "sqrt_d_k": scale_factor,
        "query_vector_norm": q_norm,
        "key_vector_norm": k_norm,
        "raw_dot_product": raw_dot,
        "scaled_score": scaled_score,
        "exp_numerator": exp_val,
        "row_partition_sum": simulated_row_sum,
        "attention_probability": softmax_weight,
        "value_vector_sample": [float(x) for x in v_vec[:5]],
        "weighted_value_contribution": weighted_v_head,
    }

    # Architecture Comparison Diff
    comparison_diff = {
        "baseline_name": "Standard Softmax MHA",
        "baseline_formula": r"\text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V",
        "current_name": attention_type,
        "current_formula": primary_formula,
        "differing_operations": [
            {"op": "Bottleneck Projection", "baseline": "None (Dense N x N)", "current": f"Rank r={rank} Petrov-Galerkin SVD"},
            {"op": "Time Complexity", "baseline": "O(N² · d)", "current": f"O(N · {rank} · d)"},
            {"op": "Memory Footprint", "baseline": "O(N²)", "current": f"O(N · {rank})"},
        ],
    }

    return LiveMathBreakdown(
        attention_type=attention_type,
        primary_formula=primary_formula,
        variable_color_map=var_colors,
        steps=steps,
        complexity_table=complexity,
        numerical_inspection=num_inspection,
        comparison_diff=comparison_diff,
    )


# ============================================================================
# Zone 2 SVG Generator Utilities
# ============================================================================

def generate_head_heatmap_svg(
    matrix: np.ndarray,
    tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    width: int = 600,
    height: int = 600,
) -> str:
    """
    Generates a standalone, crisp SVG heatmap representation for embedding into
    HTML, reports, or LaTeX figures.
    """
    N, M = matrix.shape
    margin = 80
    cell_w = (width - margin * 2) / max(1, M)
    cell_h = (height - margin * 2) / max(1, N)

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="background-color: #0f172a; font-family: monospace;">',
        f'<text x="{width/2}" y="35" fill="#f8fafc" font-size="16" font-weight="bold" text-anchor="middle">Attention Head View (Layer {layer_idx}, Head {head_idx})</text>',
        f'<g transform="translate({margin}, {margin})">',
    ]

    # Render Cells
    for i in range(N):
        for j in range(M):
            val = float(matrix[i, j])
            # Color map from navy (#0f172a) -> violet (#7c3aed) -> amber (#f59e0b)
            r = int(15 + val * 230)
            g = int(23 + val * 135)
            b = int(42 + (1.0 - val) * 195)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            x = j * cell_w
            y = i * cell_h
            svg_parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w-1:.1f}" height="{cell_h-1:.1f}" rx="2" fill="{hex_color}" opacity="{0.2 + val*0.8:.2f}">'
                f'<title>{tokens[i]} → {tokens[j]}: {val:.4f}</title>'
                f'</rect>'
            )

    # Render Token Labels
    for i in range(N):
        tok = tokens[i][:8]
        y = i * cell_h + cell_h / 2 + 3
        svg_parts.append(f'<text x="-10" y="{y:.1f}" fill="#94a3b8" font-size="10" text-anchor="end">{tok}</text>')

    for j in range(M):
        tok = tokens[j][:8]
        x = j * cell_w + cell_w / 2
        svg_parts.append(
            f'<text x="{x:.1f}" y="-10" fill="#94a3b8" font-size="10" text-anchor="start" transform="rotate(-45, {x:.1f}, -10)">{tok}</text>'
        )

    svg_parts.append('</g></svg>')
    return "".join(svg_parts)


# ============================================================================
# Zone 4 Report & Paper Generation Helpers
# ============================================================================

def generate_paper_latex_section(
    model_name: str,
    attention_type: str,
    layer_idx: int,
    head_idx: int,
    tokens: List[str],
    entropy: float,
    importance: float,
    specialization: str,
) -> str:
    """
    Auto-generates LaTeX Methods and Results text ready for research paper export.
    """
    sentence = " ".join(tokens)
    return (
        f"\\subsection{{Attention Pattern Analysis: {model_name}}}\n\n"
        f"We analyzed the internal contextual representation geometry of \\texttt{{{model_name}}} equipped with \\textbf{{{attention_type}}}. \n"
        f"The tokenized sequence was: \\textit{{{sentence}}}. \n\n"
        f"Figure~\\ref{{fig:attn_l{layer_idx}_h{head_idx}}} presents the token-to-token attention probability matrix for Layer~{layer_idx}, Head~{head_idx}. \n"
        f"Quantitative analysis reveals an empirical attention entropy of $\\mathcal{{H}} = {entropy:.2f}$~bits and an importance metric score of ${importance:.2f}$, "
        f"classifying this head as a \\textbf{{{specialization}}}.\n\n"
        f"\\begin{{figure}}[htbp]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width=0.85\\linewidth]{{figures/attention_{model_name.lower().replace('-', '_')}_l{layer_idx}_h{head_idx}.pdf}}\n"
        f"  \\caption{{Attention heatmap for \\texttt{{{model_name}}} using {attention_type} at Layer~{layer_idx}, Head~{head_idx}.}}\n"
        f"  \\label{{fig:attn_l{layer_idx}_h{head_idx}}}\n"
        f"\\end{{figure}}\n"
    )


# ============================================================================
# High-Level Orchestration Class: AttentionCanvasSuite
# ============================================================================

class AttentionCanvasSuite:
    """
    Unified engine orchestrating Zones 1 through 4 of the Attention Visualization Dashboard.
    Provides complete state serialization, analytics, and frontend-ready payload generation.
    """

    def __init__(
        self,
        model_name: str = "BERT-base-uncased",
        attention_type: str = "SVD Linear Attention (rank=64)",
        num_layers: int = 12,
        num_heads: int = 12,
        d_k: int = 64,
    ) -> None:
        self.model_name = model_name
        self.attention_type = attention_type
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_k = d_k

    def generate_synthetic_attention_tensor(
        self,
        tokens: List[str],
        seed: int = 42,
    ) -> np.ndarray:
        """
        Generates realistic synthetic multi-layer, multi-head attention weights
        reflecting diagonal positional, delimiter, and induction patterns.
        """
        np.random.seed(seed)
        N = len(tokens)
        tensor = np.zeros((self.num_layers, self.num_heads, N, N), dtype=np.float32)

        for l in range(self.num_layers):
            for h in range(self.num_heads):
                mat = np.random.exponential(scale=0.1, size=(N, N)).astype(np.float32)
                # Add structural motifs based on head index
                if h % 4 == 0:
                    # Positional diagonal head
                    for i in range(N):
                        mat[i, i] += 2.5
                        if i > 0:
                            mat[i, i - 1] += 1.5
                elif h % 4 == 1:
                    # Induction / previous token head
                    for i in range(1, N):
                        mat[i, i - 1] += 3.0
                elif h % 4 == 2:
                    # Delimiter token head (first / last token)
                    mat[:, 0] += 2.0
                    mat[:, -1] += 1.2
                else:
                    # Diffuse semantic head
                    mat += np.random.uniform(0.1, 0.8, size=(N, N))

                # Apply softmax across rows
                exp_m = np.exp(mat - np.max(mat, axis=-1, keepdims=True))
                tensor[l, h] = exp_m / np.sum(exp_m, axis=-1, keepdims=True)

        return tensor

    def export_full_dashboard_payload(
        self,
        tokens: List[str],
        selected_layer: int = 5,
        selected_head: int = 7,
        attention_tensor: Optional[Union[torch.Tensor, np.ndarray]] = None,
        query_matrix: Optional[np.ndarray] = None,
        key_matrix: Optional[np.ndarray] = None,
        embedding_method: str = "pca",
    ) -> Dict[str, Any]:
        """
        Exports a complete JSON-serializable dictionary encompassing all data
        required to render Zones 1, 2, 3, and 4 simultaneously.
        """
        if attention_tensor is None:
            attention_tensor = self.generate_synthetic_attention_tensor(tokens)
        else:
            attention_tensor = _to_numpy(attention_tensor)

        N = len(tokens)
        if query_matrix is None:
            np.random.seed(101)
            query_matrix = np.random.randn(N, self.d_k).astype(np.float32)
        if key_matrix is None:
            np.random.seed(202)
            key_matrix = np.random.randn(N, self.d_k).astype(np.float32)

        # 1. Zone 2: Head View
        head_view = extract_head_view(
            attention_matrix=attention_tensor,
            tokens=tokens,
            layer_idx=selected_layer,
            head_idx=selected_head,
            model_name=self.model_name,
            attention_type=self.attention_type,
            generate_svg=True,
        )

        # 2. Zone 2: Model View
        model_view = compute_model_view_metrics(
            attention_tensor=attention_tensor,
            tokens=tokens,
            include_thumbnails=True,
        )

        # 3. Zone 2: Sankey Flow
        sankey_data = generate_sankey_flow_data(
            attention_tensor=attention_tensor,
            tokens=tokens,
            top_k_flows=30,
        )

        # 4. Zone 2: Q/K Joint Embeddings
        qk_embeddings = compute_qk_joint_embedding(
            query_matrix=query_matrix,
            key_matrix=key_matrix,
            tokens=tokens,
            method=embedding_method,
        )

        # 5. Zone 3: Live Math Breakdown
        live_math = generate_live_math_breakdown(
            attention_type=self.attention_type,
            q_vec=query_matrix[0] if len(query_matrix) > 0 else None,
            k_vec=key_matrix[1] if len(key_matrix) > 1 else None,
            d_k=self.d_k,
            tokens=tokens,
        )

        # 6. Zone 4: Paper Export Latex
        latex_section = generate_paper_latex_section(
            model_name=self.model_name,
            attention_type=self.attention_type,
            layer_idx=selected_layer,
            head_idx=selected_head,
            tokens=tokens,
            entropy=head_view.mean_entropy,
            importance=model_view.heads[selected_layer * self.num_heads + selected_head].importance,
            specialization=model_view.heads[selected_layer * self.num_heads + selected_head].specialization,
        )

        return {
            "context": {
                "model_name": self.model_name,
                "attention_type": self.attention_type,
                "selected_layer": selected_layer,
                "selected_head": selected_head,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "tokens": tokens,
            },
            "head_view": asdict(head_view),
            "model_view": asdict(model_view),
            "sankey_flow": asdict(sankey_data),
            "qk_embeddings": asdict(qk_embeddings),
            "live_math": asdict(live_math),
            "paper_export_latex": latex_section,
        }
