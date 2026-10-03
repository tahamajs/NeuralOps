Module neuralops.core.registry
==============================
Central registry system for the Efficient Transformer Toolkit.
All framework components (attentions, compressors, trainers, visualizers, etc.)
are stored in registries, enabling dynamic discovery and plug‑and‑play integration.

Functions
---------

`get_evaluatable_modules(include_compressors: bool = True) ‑> list[str]`
:   Returns all genuine registered attention and compressor architectures, excluding helper aliases.

`get_module_capabilities(name: str) ‑> dict[str, typing.Any]`
:   Return capability flags for either attention or compressor module.

`is_helper_module(name: str) ‑> bool`
:   Returns True if the given module name corresponds to an injection helper or factory alias.

`is_module_declared_compatible(name: str, family: str) ‑> tuple[bool, str]`
:   Check whether a module's declared capabilities satisfy a model family's architectural requirements.

`list_all_components() ‑> dict`
:   Convenience function that returns all global registry keys.

Classes
-------

`Registry(name: str)`
:   A named dictionary that maps string keys to classes or factory functions with capability metadata.

    ### Methods

    `get(self, key: str) ‑> <class 'collections.abc.Callable'> | None`
    :   Safely retrieve a registered component, or None if not found.

    `get_metadata(self, key: str) ‑> dict[str, typing.Any]`
    :   Retrieve capability metadata for a registered component.

    `items(self)`
    :   Return registered key-value pairs.

    `keys(self) ‑> list[str]`
    :   Return a list of all registered keys (in original casing).

    `list(self) ‑> list[str]`
    :   Return a list of all registered keys.

    `list_available(self) ‑> list[str]`
    :   Return a list of all registered keys.

    `register(self, key: str | None = None, supports_causal: bool | None = None, supports_bidirectional: bool | None = None, supports_cross_attention: bool | None = None, supports_gqa: bool | None = None, **extra_metadata: Any) ‑> <class 'collections.abc.Callable'>`
    :   Decorator to register a class/factory under the given key with capability flags.
        
        Usage:
            @ATTENTION_REGISTRY.register("griffin", supports_causal=True, supports_bidirectional=False)
            class GriffinAttention(BaseAttention): ...

    `resolve(self, key: str) ‑> <class 'collections.abc.Callable'>`
    :   Retrieve a registered component and raise a descriptive error if missing.

    `values(self)`
    :   Return registered values.