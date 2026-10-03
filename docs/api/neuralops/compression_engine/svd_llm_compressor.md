Module neuralops.compression_engine.svd_llm_compressor
======================================================
SVD-LLM Compressor: Truncation-Aware Singular Value Decomposition for LLM Compression (ICLR 2025).
Paper Reference: Wang et al., "SVD-LLM: Truncation-Aware Singular Value Decomposition for Large Language Model Compression", ICLR 2025 (arXiv:2403.07378).

Features implemented faithfully according to the paper:
1. Data-Driven Cholesky Activation Whitening:
   - Accumulate input activation covariance C = (1/N) X^T X using calibration data
   - Cholesky factorization S = cholesky(C + eps * I)
   - Whiten weight: W_white = W @ S
   - Truncated SVD: W_white ≈ U_r @ diag(Sigma_r) @ Vt_r
   - Unwhiten right factor: Vt_unw = Vt_r @ S^{-1}
   - Explicit factor matrices: W_u = U_r * sqrt(Sigma_r), W_v = sqrt(Sigma_r) * Vt_unw
   - Forward mapping: y = (x @ W_v.T) @ W_u.T + bias
2. Sequential LoRA Fine-Tuning (Section 3.2, Eq. 7):
   - Stage 1: Update W_u via LoRA (freeze W_v)
   - Stage 2: Update W_v via LoRA (freeze W_u)
3. Fixed Parameter Rank Ratio:
   - Given parameter ratio R ∈ (0, 1]: r = (R * m * n) / (m + n)
4. Compatibility across DistilBERT, RoBERTa/BERT, GPT-2 (Conv1D), and LLaMA/Qwen.

Classes
-------

`SVDLLMCompressor(model: nn.Module, rank_ratio: float = 0.2, rank: int | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, max_error: float = 0.5, fallback_to_original: bool = False, calibration_texts: list[str] | None = None, tokenizer: Any | None = None, device: str | torch.device | None = None, target_attention_only: bool = True, recovery_steps: int = 50, learning_rate: float = 0.0002, weight_decay: float = 0.01, batch_size: int = 8, **kwargs: Any)`
:   SVD-LLM (ICLR 2025) Compressor with Cholesky activation whitening,
    sequential LoRA recovery fine-tuning, and fixed rank parameter ratios.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress the target model using full SVD-LLM Cholesky whitening and structural replacement.

    `compute_rank_from_ratio(self, m: int, n: int, ratio: float) ‑> int`
    :   Compute low rank r from parameter retention ratio R:
        Original params = m * n
        Target params = R * m * n
        Factorized params = r * (m + n)
        r = (R * m * n) / (m + n)

`SVDLLMLinear(in_features: int, out_features: int, rank: int, bias: bool = True, dtype: torch.dtype = torch.float32, device: torch.device | None = None, lora_rank: int = 16, lora_alpha: float = 32.0)`
:   SVD-LLM Factorized Linear Layer executing as:
    y = (x @ W_v.T + lora_v(x)) @ (W_u.T) + lora_u(x @ W_v.T) + bias
    
    Specifically, with sequential LoRA support:
    - W_u ∈ R^(d_out × r), W_v ∈ R^(r × d_in)
    - LoRA for W_u: A_u ∈ R^(r_L × r), B_u ∈ R^(d_out × r_L)
    - LoRA for W_v: A_v ∈ R^(r_L × d_in), B_v ∈ R^(r × r_L)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `reset_base_parameters(self) ‑> None`
    :

    `reset_lora_parameters(self) ‑> None`
    :

    `set_active_lora_stage(self, stage: str | None) ‑> None`
    :   stage == 'stage1': train A_u, B_u only (W_v frozen)
        stage == 'stage2': train A_v, B_v only (W_u frozen)
        stage == 'both': train all LoRA adapters
        stage is None: freeze all LoRA adapters

    `set_factors(self, W_u: torch.Tensor, W_v: torch.Tensor, bias: torch.Tensor | None = None) ‑> None`
    :