Module neuralops.compression_engine.sparsegpt
=============================================
SparseGPT: One-Shot Layer-Wise Weight Pruning with Second-Order Error Compensation.
Registered in COMPRESSOR_REGISTRY as 'sparsegpt'.

Mathematical Formulation:
1. Empirical Hessian Formulation:
   $$H = 2 X X^\top + \lambda I \in \mathbb{R}^{d_{in} \times d_{in}}$$
2. Cholesky Factorization & Inversion:
   $$H = L L^\top \implies H^{-1} = (L^{-1})^\top L^{-1}$$
3. OBS Saliency Metric for Column $j$:
   $$S_{ij} = \frac{|W_{ij}|^2}{2 [H^{-1}]_{jj}}$$
4. 2:4 Structured or Unstructured Mask $\mathcal{M}$:
   $$W_{pruned} = W \odot \mathcal{M}$$
5. Optimal Brain Surgeon Column Update:
   $$W_{:, j+1:d_{in}} \leftarrow W_{:, j+1:d_{in}} - (W_{:, j} - W_{pruned, :, j}) \frac{[H^{-1}]_{j, j+1:d_{in}}}{[H^{-1}]_{jj}}$$

Functions
---------

`sparsegpt_factory(model: Any, comp_config: Any = None, return_stats: bool = False, return_tuple: bool = False, **kwargs: Any) ‑> Any`
:   Factory function for SparseGPT compressor.

Classes
-------

`SparseGPTCompressor(model: nn.Module, tokenizer: Any | None = None, sparsity: float = 0.5, prunen: int = 2, prunem: int = 4, block_size: int = 128, percdamp: float = 0.01, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, target_attention_only: bool = False, device: str | torch.device | None = None, structured_24: bool = False, max_error: float = 0.5, fallback_to_original: bool = True, target_modules: list[str] | None = None, **kwargs: Any)`
:   In-memory PyTorch SparseGPT Compressor.
    
    Performs layer-by-layer second-order pruning with Optimal Brain Surgeon (OBS) weight
    compensation, supporting unstructured sparsity, block sparsity, and strict 2:4 structured sparsity.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Executes SparseGPT compression across all target linear layers.