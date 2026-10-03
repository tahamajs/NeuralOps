Module neuralops.core.model_wrapper
===================================
UnifiedTransformerModel – wraps any HuggingFace model with Hub support,
attention replacement, and config serialization.

Usage:
    model = UnifiedTransformerModel.from_pretrained("distilbert-base-uncased")
    model.replace_attention("galerkin", rank=32)
    model.save_pretrained("./compressed_model")
    # Later:
    loaded = UnifiedTransformerModel.from_pretrained("./compressed_model")

Classes
-------

`UnifiedTransformerConfig(attention_type: str = None, attention_rank: int | None = None, attention_kwargs: dict[str, typing.Any] | None = None, **kwargs)`
:   Lightweight container for the custom attention configuration.

    ### Static methods

    `from_dict(d: dict[str, typing.Any]) ‑> neuralops.core.model_wrapper.UnifiedTransformerConfig`
    :   From dict.

    ### Methods

    `to_dict(self) ‑> dict[str, typing.Any]`
    :   Serialise to a JSON‑compatible dict.

`UnifiedTransformerModel(model: torch.nn.modules.module.Module, custom_config: neuralops.core.model_wrapper.UnifiedTransformerConfig | None = None)`
:   A wrapper around a HuggingFace model that supports custom attention
    and can be saved/loaded via the HuggingFace Hub.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module
    * huggingface_hub.hub_mixin.PyTorchModelHubMixin
    * huggingface_hub.hub_mixin.ModelHubMixin

    ### Static methods

    `from_pretrained(model_id: str, task: str = 'sequence_classification', num_labels: int | None = None, **kwargs) ‑> neuralops.core.model_wrapper.UnifiedTransformerModel`
    :   Create a new UnifiedTransformerModel from a pretrained identifier.
        Automatically applies the custom attention if the model was saved with one.
        This is the main entry point for users.

    ### Methods

    `forward(self, *args, **kwargs) ‑> Any`
    :   Forward pass — compute module output from inputs.

    `freeze_layers(self, num_layers_to_freeze: int = 0) ‑> Any`
    :   Freeze layers.

    `get_parameter_count(self) ‑> dict[str, int]`
    :   Return the parameter count.

    `replace_attention(self, attention_type: str, rank: int | None = None, copy_weights: bool = True, **kwargs) ‑> Any`
    :   Inject a custom attention variant into all layers.
        This method also records the choice so it can be saved.

    `save_pretrained(self, save_directory: str) ‑> None`
    :   Convenience wrapper – delegates to the mixin.

    `unfreeze_all(self) ‑> Any`
    :   Unfreeze all.