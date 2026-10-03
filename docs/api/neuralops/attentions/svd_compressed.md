Module neuralops.attentions.svd_compressed
==========================================
SVD-Compressed Attention with Importance-Aware Lagrangian Rank Allocation.

Truncated Singular Value Decomposition (SVD) attention compression module with
Lagrangian spectral-entropy optimal rank budgeting and optional low-rank plus sparse residual decomposition.

Mathematical Mechanism:
    For projection weight matrices $\{W_i\}_{i=1}^M$ across all transformer attention sub-layers:
    1. **Spectral Information Loss & Parameter Cost:**
       $$E_i(r) = \frac{\sum_{j=1}^r \sigma_{i, j}^2}{\sum_{j=1}^{\min(d_{\text{in}}, d_{\text{out}})} \sigma_{i, j}^2}, \quad \operatorname{Cost}_i(r) = 1 - E_i(r)$$
       $$\operatorname{Params}_i(r) = r \cdot (d_{i, \text{in}} + d_{i, \text{out}})$$
    2. **Lagrangian Global Rank Optimization:**
       $$\min_{\{r_i\}} \sum_{i=1}^M \operatorname{Cost}_i(r_i) \quad \text{subject to} \quad \sum_{i=1}^M \operatorname{Params}_i(r_i) \le B_{\text{target}}$$
    3. **Low-Rank Factorized Multi-Head Projections:**
       $$W_i \approx W_{i}^{\text{up}} W_{i}^{\text{down}}, \quad W_{i}^{\text{down}} \in \mathbb{R}^{r_i \times d_{\text{in}}}, \, W_{i}^{\text{up}} \in \mathbb{R}^{d_{\text{out}} \times r_i}$$
       $$\operatorname{Attn}(Q, K, V) = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M\right) V$$

Computational Complexity:
    - Low-rank projection per token: $\mathcal{O}(d_{\text{in}} r_i + r_i d_{\text{out}})$.
    - Attention matrix evaluation: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d)$.
    - Total per-layer complexity: $\mathcal{O}(B \cdot N \cdot (d_{\text{in}} r + r d_{\text{out}}) + B \cdot H \cdot N^2 \cdot d)$.

Functions
---------

`apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor | None = None, unsqueeze_dim: int = 1) ‑> tuple[torch.Tensor, torch.Tensor]`
:   Applies Rotary Position Embedding (RoPE) with partial dimension and independent seq-len support.

`calculate_spectral_entropy(singular_values: numpy.ndarray) ‑> float`
:   Compute normalized spectral entropy of singular value distribution.

`compute_cumulative_entropy(singular_values: numpy.ndarray, max_rank: int) ‑> numpy.ndarray`
:   Compute cumulative spectral energy fraction retained by top-$k$ singular values.

`compute_metrics(eval_pred: Any) ‑> dict[str, float]`
:   Compute classification evaluation accuracy.

`gather_all_attention_weights(model: Any) ‑> tuple[list[numpy.ndarray], list[str]]`
:   Gather attention projection weights and layer names across model layers.

`get_num_labels_and_is_regression(task: str) ‑> tuple[int, bool]`
:   Return class count and regression indicator for GLUE task.

`inject_all_layers(model: Any = None, rank_dict: dict[str, int] | None = None, dropout: float = 0.1, add_sparse: bool = False, sparsity: float = 0.0, config: Any = None, **kwargs) ‑> Any`
:   Inject SVDCompressedAttention into all transformer layers of model.

`inject_svd_compressed(model: torch.nn.modules.module.Module | None = None, layer: torch.nn.modules.module.Module | None = None, model_config: Any | None = None, comp_config: Any | None = None, layer_idx: int = 0, total_layers: int = 12, old_attn: Any | None = None, lora_rank: int | None = None, **kwargs: Any) ‑> neuralops.attentions.svd_compressed.SVDCompressedAttention | torch.nn.modules.module.Module | None`
:   Bridge helper safely injecting SVDCompressedAttention into a transformer layer or model.

`lagrangian_rank_allocation(weight_matrices: list[numpy.ndarray], target_budget: int, layer_proj_names: list[str], max_rank_factor: float = 0.5, lambda_bracket: tuple[float, float] = (1e-05, 10.0), num_lambda_steps: int = 100) ‑> dict[str, int]`
:   Allocate per-projection ranks under global parameter constraint via binary search on Lagrange multiplier.

`load_glue_local(task_name: str) ‑> Any`
:   Load GLUE benchmark dataset from local offline storage.

`main() ‑> None`
:   Run full training workflow for SVDCompressedAttention on GLUE.

`preprocess_function(examples: Any, tokenizer: Any, sent1_key: str, sent2_key: str | None, max_length: int, is_regression: bool) ‑> dict[str, typing.Any]`
:   Preprocess and tokenize dataset sequence examples.

Classes
-------

`SVDCompressedAttention(config: Any | None = None, old_attn: Any | None = None, dropout: float = 0.1, layer_idx: int = 0, ranks: dict[str, int] | None = None, lora_rank: int = 16, lora_alpha: float = 32.0, is_decoder: bool | None = None, add_sparse: bool = False, sparsity_level: float = 0.7, **kwargs: Any)`
:   Multi-head attention with truncated SVD projections, trainable LoRA adapters,
    GQA support, RoPE support, and optional sparse residuals.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseAttention
    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    `supports_gqa: bool`
    :   The type of the None singleton.

    ### Methods

    `copy_weights(self, old_attn: Any) ‑> None`
    :   Robust weight copying from DistilBERT, RoBERTa nested, GPT-2 Conv1D, or Qwen/LLaMA.