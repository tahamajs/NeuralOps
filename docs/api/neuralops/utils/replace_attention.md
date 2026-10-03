Module neuralops.utils.replace_attention
========================================

Functions
---------

`replace_attention(model, attention_key, layer_indices=None, **kwargs) ‑> Any`
:   Replace attention modules in a model with a registered attention block.
    
    Args:
        model: A PyTorch model (typically a Hugging Face transformer)
        attention_key: String key registered in ATTENTION_REGISTRY
        layer_indices: Optional list of layer indices to replace (0-indexed).
                       If None, replace all attention layers.
        **kwargs: Additional arguments passed to the attention block constructor.
    
    Returns:
        Model with attention layers replaced.

Classes
-------

`UniversalAttentionWrapper(inner_attn: torch.nn.modules.module.Module, layer_name: str = '', return_tuple: bool | None = None)`
:   Universal forward adapter wrapping any custom attention module.
    Seamlessly adapts across all HuggingFace architectures:
      - DistilBERT: forward(query=x, key=x, value=x, mask=mask, head_mask=head_mask, output_attentions=...) -> tuple (out, weights)
      - BERT / RoBERTa: forward(hidden_states, attention_mask, ...) -> tuple (out, weights)
      - LLaMA / Mistral: forward(hidden_states, attention_mask, position_ids, ...) -> tuple (out, weights, ...)
      - GPT-2: forward(x, layer_past, ...) -> tuple (out, present)
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `inner: torch.nn.modules.module.Module`
    :

    ### Methods

    `forward(self, *args: Any, **kwargs: Any) ‑> tuple[torch.Tensor, torch.Tensor | None]`
    :   Define the computation performed at every call.
        
        Should be overridden by all subclasses.
        
        .. note::
            Although the recipe for forward pass needs to be defined within
            this function, one should call the :class:`Module` instance afterwards
            instead of this since the former takes care of running the
            registered hooks while the latter silently ignores them.