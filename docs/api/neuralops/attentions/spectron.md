Module neuralops.attentions.spectron
====================================
SpectronAttention: Importance-Aware Low-Rank + Sparse Attention (Activation-Aware v2).

Dynamically allocates non-uniform low-rank capacities across attention projection
matrices under a global parameter budget via Lagrangian spectral entropy
optimization, augmented by trainable magnitude-pruned sparse residuals.

Mathematical Mechanism:
    For projection matrices {W_i} with singular values {σ_ik}:
    1. Lagrangian Spectral Allocation (on ACTIVATION-WHITENED weights):
         min_{r_i} Σ (1 - Σ_{k≤r_i} σ²_ik / Σ_k σ²_ik)
         s.t.     Σ r_i (d_in + d_out) ≤ B_target
       Solved via bisection on λ.

    2. Activation-Aware Factorization:
         C_X = E[x x^T]                  (EMA covariance)
         C_X = L L^T                     (Cholesky whitening)
         W_w = W L
         U Σ V^T = SVD(W_w)
         W ≈ (U_r Σ_r) (L^{-1} V_r^T)    (whitened low-rank)

    3. Optional Sparse Residual:
         y = x W_down^T W_up^T + x S^T + b
         with ‖S‖_0 ≤ (1 - ρ) · numel(W)

Functions
---------

`allocate_ranks_spectron(weight_matrices: list[np.ndarray], target_budget: int, names: list[str], max_rank_factor: float = 0.75, lambda_range: tuple[float, float] = (1e-05, 5.0), steps: int = 100, whitening_factors: list[np.ndarray | None] | None = None) ‑> dict[str, int]`
:   Lagrangian optimization allocating rank per weight matrix under a
    parameter budget.
    
    SP-FIX-01: if `whitening_factors` is given, the SVD is computed on the
    whitened weight W @ L, so the allocation reflects the *activation-aware*
    spectral profile — not the raw weight spectrum.

`cumulative_entropy_fraction(s: np.ndarray, max_rank: int) ‑> numpy.ndarray`
:   

`gather_attention_weights(model: Any) ‑> tuple[list[numpy.ndarray], list[str]]`
:   Return flattened list of Q/K/V/O weight matrices and their layer names.
    Supports DistilBERT, BERT/RoBERTa, GPT-2, LLaMA/Qwen.

`inject_spectron(layer: Any = None, model: Any = None, model_config: Any = None, comp_config: Any = None, layer_idx: int = 0, total_layers: int = 12, old_attn: nn.Module | None = None, **kwargs: Any)`
:   

`spectral_entropy(singular_values: np.ndarray) ‑> float`
:   

Classes
-------

`LowRankSparseLinear(down: nn.Linear, up: nn.Linear, sparse_weight: torch.Tensor, bias: torch.Tensor | None = None)`
:   y = up(down(x)) + F.linear(x, S) + b
    
    S is a trainable sparse residual. Its L1 gradient encourages zero
    entries; periodic `prune_sparse` enforces hard threshold.
    
    SP-FIX-02: bias stored as a proper nn.Parameter.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

    `prune_sparse(self, sparsity: float) ‑> None`
    :   Hard threshold: keep only top (1 - sparsity) fraction of entries.

    `sparse_l1_loss(self) ‑> torch.Tensor`
    :

`SparsePruningCallback(model: Any, prune_freq: int = 200, final_sparsity: float = 0.7)`
:   Periodically hard-thresholds the sparse residual of every
    SpectronAttention module in the model.

    ### Ancestors (in MRO)

    * transformers.trainer_callback.TrainerCallback

    ### Methods

    `on_step_end(self, args, state, control, **kwargs)`
    :   Event called at the end of a training step. If using gradient accumulation, one training step might take
        several inputs.

`SpectronAttention(config: Any | None = None, old_attn: Any | None = None, dropout: float = 0.1, layer_idx: int = 0, ranks: dict[str, int] | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, add_sparse: bool = False, sparsity_level: float = 0.7, whitening: dict[str, torch.Tensor] | None = None, seed: int = 42, **kwargs: Any)`
:   Every custom attention module must inherit from this.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_causal: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Methods

    `prune_all_sparse(self, sparsity: float) ‑> None`
    :

    `total_sparse_l1(self) ‑> torch.Tensor`
    :   Sum of L1 norms of all sparse residuals. Add to training loss.