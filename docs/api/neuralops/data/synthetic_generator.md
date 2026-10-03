Module neuralops.data.synthetic_generator
=========================================
High-Fidelity Synthetic Dataset & Token Generator for NLP / LLM Evaluation.

Replaces uniform random token noise with a power-law Zipfian distribution:
$$P(k) = \frac{1 / k^s}{\sum_{n=1}^V 1 / n^s} \quad \text{with } s \approx 1.15$$
to simulate authentic natural language entropy, word rank frequency curves,
and valid causal perplexity evaluation dynamics.

Functions
---------

`compute_empirical_entropy(tokens: torch.Tensor, vocab_size: int = 32000) ‑> float`
:   Computes the empirical Shannon entropy (in bits) of a tokenized tensor.
    
    Args:
        tokens: Long tensor of token IDs.
        vocab_size: Total vocabulary size.
    
    Returns:
        float: Shannon entropy $H(X) = -\sum p_i \log_2(p_i)$ in bits.

`generate_synthetic_data(output_path: str, num_samples: int = 10000, domain: str = 'movies', seed: int = 42) ‑> pandas.DataFrame`
:   Generate a single-domain synthetic classification dataset.

`generate_zipfian_markov_tokens(num_samples: int, seq_len: int, vocab_size: int = 32000, zipf_param: float = 1.15, transition_temp: float = 0.8, seed: int = 42, min_token_id: int = 1) ‑> torch.Tensor`
:   Generates token sequences with Zipfian marginal rank distribution combined with
    first-order Markov sequential transition coupling.
    
    Args:
        num_samples: Number of sequences.
        seq_len: Sequence length.
        vocab_size: Vocabulary size.
        zipf_param: Zipf exponent.
        transition_temp: Softmax temperature for Markov transition matrix.
        seed: Random seed.
        min_token_id: Minimum token ID offset.
    
    Returns:
        torch.Tensor: Long tensor of shape `(num_samples, seq_len)`.

`generate_zipfian_tokens(num_samples: int, seq_len: int, vocab_size: int = 32000, zipf_param: float = 1.15, seed: int = 42, min_token_id: int = 1) ‑> torch.Tensor`
:   Generates token sequences following a Zipfian power-law distribution.
    
    Ensures that low-index tokens (common stop words, articles) occur with high frequency
    while high-index tokens (rare technical terms) occur sparsely, mirroring natural language.
    
    Args:
        num_samples: Number of sequences to generate.
        seq_len: Sequence length (number of tokens per sample).
        vocab_size: Size of vocabulary.
        zipf_param: Power-law exponent $s > 0$ (default $1.15$, typical for natural language).
        seed: Random seed for deterministic reproducibility.
        min_token_id: Minimum token ID offset (default 1 to reserve 0 for PAD).
    
    Returns:
        torch.Tensor: Long tensor of shape `(num_samples, seq_len)` containing sampled token IDs.

`run_universal_generation(output_dir: str | Path, total_samples_per_domain: int = 5000, domains: dict[str, Any] | None = None, seed: int = 42) ‑> pandas.DataFrame`
:   Generate synthetic dataset across multiple domains and combine them.

Classes
-------

`SyntheticClassificationDataset(num_samples: int = 64, seq_len: int = 32, vocab_size: int = 1000, num_classes: int = 2, seed: int = 42)`
:   High-fidelity synthetic classification dataset using Zipfian token distributions
    and class-conditional semantic bias for realistic classifier benchmarking.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

`SyntheticPerplexityDataset(num_samples: int = 128, seq_len: int = 128, vocab_size: int = 32000, zipf_param: float = 1.15, seed: int = 42)`
:   Synthetic Language Modeling Dataset using Zipfian power-law token distribution
    for authentic causal perplexity evaluation without empirical artifact distortion.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic

`SyntheticLanguageModelingDataset(num_samples: int = 128, seq_len: int = 128, vocab_size: int = 32000, zipf_param: float = 1.15, seed: int = 42)`
:   Synthetic Language Modeling Dataset using Zipfian power-law token distribution
    for authentic causal perplexity evaluation without empirical artifact distortion.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic