Module neuralops.compression_engine.flat_llm
============================================
FLAT-LLM: Activation-Aware Structured Dimension & Neuron Pruning.

Registered as 'flat_llm', 'flat_llm_compressor'.

Functions
---------

`flat_llm_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   

Classes
-------

`FLATLLMCompressor(model: nn.Module, tokenizer: Any | None = None, keep_ratio: float = 0.7, damping: float = 0.0001, max_error: float = 0.35, target_attention_only: bool = False, fallback_to_original: bool = True, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, structured_24: bool = False, target_modules: list[str] | None = None, cfg: Any | None = None, num_calibration_batches: int = 8, **kwargs: Any)`
:   FLAT-LLM compressor: Activation-aware output-channel pruning.
    
    Pipeline:
      1. Collect per-output-channel energy from calibration activations.
      2. Rank channels by energy.
      3. Keep top-k channels (k = round(keep_ratio * out_features)).
      4. Replace Linear with PrunedChannelLinear (factorized storage).
    
    Optionally applies 2:4 structured sparsity (compute-only; no storage savings
    unless a sparse kernel is used).

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

`PrunedChannelLinear(in_features: int, original_out_features: int, kept_idx: torch.Tensor, W_kept: torch.Tensor, bias_kept: torch.Tensor | None = None)`
:   Linear layer with pruned output channels.
    
    Stores only the kept rows of the effective weight matrix:
      - W_kept:      (k, in_features)  fp16/fp32   -- only kept rows
      - bias_kept:   (k,)              optional
      - kept_idx:    (k,)              int64  -- output positions of kept rows
      - original_out_features: int
    
    Forward:
      h = F.linear(x, W_kept, bias_kept)          # (..., k)
      out = zeros(..., original_out_features)
      out.scatter_(-1, kept_idx, h)               # (..., original_out)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Class variables

    `is_custom_attention`
    :   The type of the None singleton.

    ### Instance variables

    `weight: torch.Tensor`
    :   Reconstruct effective (original_out, in) weight with zeros in pruned rows.

    ### Methods

    `extra_repr(self) ‑> str`
    :   Return the extra representation of the module.
        
        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.