Module neuralops.compression_engine.mgie_svd_compressor
=======================================================
MGIE-SVD with Total Rank Budget: Global Entropy-Proportional Rank Allocation.

Registered as 'mgie_budget', 'mgie_budget_factory', 'mgie_svd_budget_factory'.

Functions
---------

`mgie_budget_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   MGIE budget factory function.

Classes
-------

`MGIESVDBudgetCompressor(model: nn.Module, total_rank_budget: int = 1000, rank: int | None = None, max_error: float = 0.5, fallback_to_original: bool = True, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = False, target_attention_only: bool = False, min_rank: int = 4, target_modules: list[str] | None = None, **kwargs: Any)`
:   MGIE-SVD Budget Compressor: Global Entropy-Proportional Rank Allocation.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.