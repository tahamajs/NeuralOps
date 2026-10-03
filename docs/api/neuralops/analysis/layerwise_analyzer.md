Module neuralops.analysis.layerwise_analyzer
============================================
Streaming Layer-wise Analysis Engine for Foundation Models & Transformers.

Computes representation drift, cosine similarities, kurtosis-based outlier metrics,
effective spectral rank, and intrinsic dimensionality on-the-fly without accumulating
all 32+ layer activations simultaneously in VRAM, eliminating streaming OOM crashes.

Functions
---------

`estimate_intrinsic_dimension(hidden_states: np.ndarray, k: int = 5) ‑> float`
:   Estimates the intrinsic dimension of a representation manifold using k-NN distances.
    
    Args:
        hidden_states: Sample representation array of shape [N, D].
        k: Neighbor rank for distance estimation.
    
    Returns:
        float: Estimated intrinsic dimension between 1.0 and D.

Classes
-------

`LayerwiseAnalyzer(model: nn.Module, tokenizer: Any, device: str | torch.device | None = None)`
:   Streaming Layerwise Analyzer that captures inter-layer representations,
    outlier distribution spikes, and effective rank without accumulating VRAM memory.

    ### Methods

    `analyze(self, texts: list[str], batch_size: int = 8) ‑> dict[str, typing.Any]`
    :   Executes streaming layerwise analysis on input text corpus.
        
        Args:
            texts: List of input string prompts/sentences.
            batch_size: Batch size for tokenization and model inference.
        
        Returns:
            Dict containing cosine similarities, effective ranks, outlier statistics,
            and least-compressible layer rankings.

    `register_streaming_hooks(self) ‑> None`
    :   Registers streaming forward hooks on all model transformer layers.

    `remove_hooks(self) ‑> None`
    :   Remove all active hooks and free memory.

`StreamingLayerAccumulator(total_tokens: int = 0, total_batches: int = 0, running_mean_vector: np.ndarray | None = None, max_activation: float = 0.0, sum_outlier_count_6sigma: int = 0, total_elements: int = 0, sum_m2: float = 0.0, sum_m4: float = 0.0, cov_matrix: np.ndarray | None = None, cov_samples: int = 0)`
:   Online Welford & spectral accumulator for single-layer activations.
    Updates statistical moments dynamically and frees raw activation buffers immediately.

    ### Instance variables

    `cov_matrix: numpy.ndarray | None`
    :   The type of the None singleton.

    `cov_samples: int`
    :   The type of the None singleton.

    `max_activation: float`
    :   The type of the None singleton.

    `running_mean_vector: numpy.ndarray | None`
    :   The type of the None singleton.

    `sum_m2: float`
    :   The type of the None singleton.

    `sum_m4: float`
    :   The type of the None singleton.

    `sum_outlier_count_6sigma: int`
    :   The type of the None singleton.

    `total_batches: int`
    :   The type of the None singleton.

    `total_elements: int`
    :   The type of the None singleton.

    `total_tokens: int`
    :   The type of the None singleton.

    ### Methods

    `compute_effective_rank(self) ‑> float`
    :   Computes the effective rank erank(Sigma) = exp(H(p)) where p_i = lambda_i / sum(lambda).

    `compute_kurtosis(self) ‑> float`
    :   Compute excess kurtosis: E[(x-mu)^4] / (E[(x-mu)^2])^2 - 3.

    `compute_outlier_ratio(self) ‑> float`
    :   Compute 6-sigma outlier fraction.

    `update(self, act_tensor: torch.Tensor) ‑> None`
    :   Dynamically updates moments from an incoming activation batch [B, S, D] or [N, D]
        and releases tensor references.