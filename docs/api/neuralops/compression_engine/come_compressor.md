Module neuralops.compression_engine.come_compressor
===================================================
CoMe: Compress LLMs through Layer Concatenation and Shedding with Learnable
Inter-Layer Interpolation Adapters.

Registered in COMPRESSOR_REGISTRY as 'come' and 'come_compressor'.

Mathematical Formulation:
1. Layer Shedding & Uniform Strided Selection:
   I_keep = {i_0, i_1, ..., i_{K-1}}, i_k = round(k * (L-1) / (K-1))

2. Learnable Low-Rank Interpolation Bridge between Non-Consecutive Layers:
   W_bridge = I + B @ A    (B: hidden x r, A: r x hidden)
   h_bridged = h + alpha * (W_bridge h - h)
             = h + alpha * (B @ A) h
   Initialized with B = 0 so bridge is identity at init.

Functions
---------

`come_factory(model: Any, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   Factory function for the CoMe compressor.

Classes
-------

`CoMeBridge(hidden_size: int, bridge_rank: int = 64, alpha_init: float = 0.1)`
:   Learnable low-rank interpolation adapter bridging representation gaps
    between non-consecutive layers.
    
        W_bridge = I + B @ A     (B: hidden x r, A: r x hidden)
        y        = x + alpha * (W_bridge x - x)
                 = x + alpha * (B @ A) x
    
    At initialization B = 0, so the bridge is exactly the identity and only
    alpha influences the output; gradient flows to A and alpha from step 1.
    Robustly handles Tensor, tuple, list, dict, and BaseModelOutput returns.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, hidden_states: Any, *args: Any, **kwargs: Any) ‑> Any`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`CoMeBridgedLayer(layer: nn.Module, bridge: CoMeBridge | None = None)`
:   Wraps a retained transformer layer with a learnable CoMeBridge.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, hidden_states: Any = None, *args: Any, **kwargs: Any) ‑> Any`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.

`CoMeCompressor(model: nn.Module, tokenizer: Any | None = None, target_layers: int | None = None, sparsity: float = 0.1, bridge_rank: int | None = None, alpha_init: float = 0.1, lora_r: int = 16, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, protect_critical: bool = True, target_attention_only: bool = False, recalibrate_ln: bool = True, allow_forward_failure: bool = False, device: str | torch.device | None = None, target_modules: list[str] | None = None, rank: int | None = None, **kwargs: Any)`
:   CoMe Compressor: Layer shedding with learnable low-rank interpolation
    adapters (bridges) between non-consecutive retained layers.
    
    Config parameters (via kwargs or comp_config):
        rank                  : int, optional — SVD-style rank. If provided and
                                bridge_rank/sparsity are not explicitly set,
                                they are derived from rank via safe mappings.
        target_layers         : int, optional — target layer count after shedding.
        sparsity              : float, default 0.10 — fraction of layers to remove.
        bridge_rank           : int, optional — bottleneck rank for the
                                low-rank interpolation bridge (B @ A). If not
                                set, derived from rank (rank // 4) or defaults
                                to 64 for the full hidden size.
        alpha_init            : float, default 0.1 — initial bridge mixing coeff.
        lora_r                : int, default 16 — post-shedding LoRA rank.
        lora_alpha            : float, default 1.0 — LoRA scaling.
        target_attention_only : bool, default False.
        skip_attention_qkv    : bool, default True — protect asymmetric GQA.
        recalibrate_ln        : bool, default True.
        allow_forward_failure : bool, default False.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   The type of the None singleton.

    ### Methods

    `get_trainable_params(self) ‑> list[torch.nn.parameter.Parameter]`
    :

`CoMeLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer: frozen base weights + optional trainable LoRA adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.come_compressor.CoMeLoRALinear`
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