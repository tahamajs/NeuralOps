Module neuralops.compression_engine.soka_compressor
===================================================
SoKA: Second-Order Knowledge-Aware Low-Rank Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'soka'.

Mathematical Formulation:
1. Weight Matrix Factorization via SVD:
   $$W = U \Sigma V^\top \approx U_r \Sigma_r V_r^\top = A B^\top$$
   where $A = U_r \Sigma_r^{1/2} \in \mathbb{R}^{d_{out} \times r}$ and $B = V_r \Sigma_r^{1/2} \in \mathbb{R}^{d_{in} \times r}$.
2. Forward Execution:
   $$y = (x B) A^\top + b$$
3. Singular Energy Threshold Rank Selection:
   $$r = \min \left\{ k \;\middle|\; \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^{\min(d_{in}, d_{out})} \sigma_i^2} \ge \alpha \right\}$$

Functions
---------

`soka_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> Any`
:   Factory function for SoKA compressor.

Classes
-------

`LowRankCompressor(model: nn.Module, rank: int = 8, energy_threshold: float | None = None, max_error: float = 0.5, target_attention_only: bool = False, fallback_to_original: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SoKA Low-Rank Linear Compressor.
        Replaces dense nn.Linear and Conv1D layers with factorized LowRankLinear layers.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Replaces target Linear and Conv1D modules with LowRankLinear.

`SoKACompressor(model: nn.Module, rank: int = 8, energy_threshold: float | None = None, max_error: float = 0.5, target_attention_only: bool = False, fallback_to_original: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   SoKA Low-Rank Linear Compressor.
        Replaces dense nn.Linear and Conv1D layers with factorized LowRankLinear layers.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Replaces target Linear and Conv1D modules with LowRankLinear.

`LowRankLinear(in_features: int, out_features: int, rank: int, bias: bool = True, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Factorized Linear Layer executing as y = (x @ V) @ U^T + b.
    Reduces parameter storage from (d_in * d_out) to r * (d_in + d_out).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstructed effective weight matrix for compatibility.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.