Module neuralops.utils.lora
===========================
Native, zero-dependency LoRA (Low-Rank Adaptation) module for PyTorch models.

Provides standalone low-rank adapters for linear layers without requiring external
PEFT packages. Supports Qwen, BERT, DistilBERT, RoBERTa, GPT-2, and T5 architectures.

Functions
---------

`apply_lora(model: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.05, target_modules: Iterable[str] | str | None = None) ‑> tuple[torch.nn.modules.module.Module, int]`
:   Freeze base model parameters and inject LoRALinear into matching target modules.
    
    Args:
        model: PyTorch module (e.g. Qwen2, DistilBERT, RoBERTa, GPT-2).
        r: LoRA rank dimension.
        alpha: LoRA scaling factor (scaling = alpha / r).
        dropout: Dropout probability applied to LoRA adapter input.
        target_modules: Iterable or comma-separated string of target submodule names.
    
    Returns:
        Tuple of (model, number_of_adapted_linear_layers).

`find_all_linear_submodule_names(model: nn.Module) ‑> set[str]`
:   Dynamically extract all Linear/Conv1D layer leaf names from model, excluding task heads.

`get_lora_trainable_parameters(model: nn.Module) ‑> list[torch.nn.parameter.Parameter]`
:   Return all parameters that require gradients (LoRA adapters + task heads).

`load_lora_weights(model: nn.Module, load_path: str, strict: bool = False) ‑> dict`
:   Load trainable parameters from saved checkpoint into model.

`save_lora_weights(model: nn.Module, save_path: str, metadata: dict | None = None) ‑> None`
:   Save only trainable parameters (LoRA weights + task head) to disk.

Classes
-------

`LoRALinear(linear: nn.Module, r: int = 16, alpha: float = 32.0, dropout: float = 0.05)`
:   Linear layer wrapped with Low-Rank Adaptation (LoRA) adapters.
    
    Initialize internal Module state, shared by both nn.Module and ScriptModule.

    ### Ancestors (in MRO)

    * torch.nn.modules.module.Module

    ### Instance variables

    `bias: torch.Tensor | None`
    :   Expose base layer bias for compatibility with hooks and inspectors.

    `weight: torch.Tensor`
    :   Expose base layer weight for compatibility with hooks and inspectors.

    ### Methods

    `forward(self, x: torch.Tensor) ‑> torch.Tensor`
    :   Forward pass combining base projection with low-rank adapter delta.

    `merge(self) ‑> None`
    :   In-place merge LoRA delta weight into the base linear layer.

    `reset_parameters(self) ‑> None`
    :   Initialize lora_A with Kaiming uniform and lora_B with zeros.

    `unmerge(self) ‑> None`
    :   Subtract LoRA delta weight to revert the base linear layer to unmerged state.