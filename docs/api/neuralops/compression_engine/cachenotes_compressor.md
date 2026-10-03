Module neuralops.compression_engine.cachenotes_compressor
=========================================================
CacheNotes: Task-Aware Key-Value Cache Compression and Static Weight Compression
with Learnable Continuous CPT (Compression-Planning-Token) Generator.

Given a task description, CacheNotes generates Compression-Planning-Tokens (CPTs)
that guide the compression of the corpus and KV cache into compact, reusable states.

Scientific Enhancement:
- Replaces static keyword heuristics with a continuous, differentiable CPTGenerator network.
- Learnable task representation using tokenized embedding bag and multi-frequency coordinate positional projection.
- Capable of evaluating task importance at ANY continuous output feature dimension d (64, 128, 768, 1024, 2048, 4096, etc.).
- Backprop-trainable end-to-end minimizing Frobenius reconstruction loss under CPT-guided low-rank SVD.
- Full fallback compatibility with static CPT heuristic if generator is untrained or disabled.

Functions
---------

`cachenotes_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) ‑> Any`
:   CacheNotes compressor factory function.

Classes
-------

`CPTGenerator(hidden_dim: int = 128, num_freqs: int = 8, vocab_size: int = 5000)`
:   Learnable Task-Aware Compression Planning Token (CPT) Generator.
    Maps arbitrary task descriptions to continuous per-feature importance vectors for weight SVD
    and KV cache compression across arbitrary dimension sizes d.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Methods

    `forward(self, task: str, d: int, device: torch.device | None = None) ‑> torch.Tensor`
    :   Generate a per-feature importance modulation vector of size d in [0.1, 2.0].

`CacheNotesCompressor(model: nn.Module, task_type: str = 'qa', task_description: str | None = None, use_learned_cpt: bool = True, keep_ratio: float = 0.5, max_error: float = 0.35, fallback_to_original: bool = True, protect_critical: bool = True, target_attention_only: bool = False, lora_r: int = 0, lora_alpha: float = 1.0, skip_attention_qkv: bool = True, recalibrate_ln: bool = True, tokenizer: Any | None = None, calibration_loader: Any | None = None, calibration_texts: list[str] | None = None, device: str | torch.device | None = None, target_modules: list[str] | None = None, **kwargs: Any)`
:   Base class for all compressors. Not a nn.Module to avoid parameter storage.

    ### Ancestors (in MRO)

    * neuralops.core.base.BaseCompressor

    ### Class variables

    `is_custom_attention: bool`
    :   CacheNotes Compressor: Decouples static weight compression from KV cache dynamic pruning
        with continuous learnable CPT Generator.
        
        Production hardening features:
        - Trainable CPTGenerator mapping task descriptions to per-feature importance vectors.
        - Support for LoRALinear wrappers, nn.Linear, and Conv1D (GPT-2).
        - Preserves existing LoRALinear wrappers without creating redundant layers.
        - target_attention_only parameter skipping FFN/MLP and preserving classification heads.
        - Robust config handling (_cfg_get) supporting None, dict, and namespace objects.
        - Fallback freezing: parameters in any fallback layer are frozen (requires_grad=False).
        - Checkpoint saving and loading for CPTGenerator.
        - Comprehensive metric tracking: actual ranks, reconstruction error, and KV cache saving.

    ### Methods

    `compress(self, **kwargs: Any) ‑> tuple[torch.nn.modules.module.Module, dict[str, typing.Any]]`
    :   Compress linear layers using task-guided CPT-weighted low-rank SVD.

    `get_importance(self, d: int, device: torch.device) ‑> torch.Tensor`
    :   Retrieves feature importance vector using learned CPT generator if trained/active,
        otherwise falling back to static heuristic.

    `load_cpt_generator(self, path: str | Path) ‑> bool`
    :   Load CPT generator state dict from file.

    `save_cpt_generator(self, path: str | Path) ‑> None`
    :   Save CPT generator state dict to file.

    `train_cpt_generator(self, task_descriptions: list[str] | None = None, sample_layers: list[tuple[str, nn.Module]] | None = None, epochs: int = 5, lr: float = 0.0001) ‑> float`
    :   Train CPT generator on sample layers across diverse task descriptions,
        minimizing reconstruction error after CPT-weighted SVD compression.

`CacheNotesKVCache(max_cache_len: int = 128, task_type: str = 'qa', keep_ratio: float = 0.3, num_sinks: int = 4, cpt_generator: CPTGenerator | None = None)`
:   Runtime task-guided dynamic KV cache compressor.
    Evaluates token task-relevance dynamically and prunes non-essential context tokens,
    supporting variable sequence lengths during prefill and autoregressive decoding.

    ### Methods

    `reset(self) ‑> None`
    :   Clear cache state to prevent memory leaks across sequences.

    `update_and_compress(self, new_k: torch.Tensor, new_v: torch.Tensor, is_new_sequence: bool = False) ‑> tuple[torch.Tensor, torch.Tensor]`
    :   Append new KV tokens and apply task-aware retention when exceeding capacity.
        Shapes: (B, H, N, D).
        Automatically resets cache on new multi-token prompt prefill (N > 1) or when is_new_sequence=True (H6).

`CacheNotesLoRALinear(in_features: int, out_features: int, weight: torch.Tensor, bias: torch.Tensor | None = None, lora_r: int = 0, lora_alpha: float = 1.0, device: torch.device | None = None, dtype: torch.dtype | None = None)`
:   Linear layer containing CacheNotes compressed base weights (frozen)
    with optional trainable LoRA adapters (lora_A, lora_B).
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Static methods

    `from_linear(linear: nn.Module, weight: torch.Tensor, lora_r: int = 0, lora_alpha: float = 1.0) ‑> neuralops.compression_engine.cachenotes_compressor.CacheNotesLoRALinear`
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

`CompressionPlanningTokens()`
:   Static CPT heuristic importance generator.
    Serves as safe baseline fallback when learnable CPT generator is inactive.

    ### Class variables

    `TASK_KEYWORD_MAP`
    :   The type of the None singleton.

    ### Static methods

    `generate_importance(task_type: str, num_features: int, device: torch.device = device(type='cpu')) ‑> torch.Tensor`
    :   Generate per-feature importance modulation vector based on static heuristics.