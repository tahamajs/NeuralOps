Module neuralops.trainer.methods.registry
=========================================

Functions
---------

`available_methods() ‑> Any`
:   Return list of all registered method names.

`get_method(name: str) ‑> Any`
:   Return the method function by name, or None if not found.

`register_method(name: str) ‑> Any`
:   Decorator to register a method callable.

Classes
-------

`MethodRegistryDict(*args, **kwargs)`
:   Custom dictionary registry with resolve() method for training methods.

    ### Ancestors (in MRO)

    * builtins.dict

    ### Methods

    `resolve(self, name: str) ‑> Any`
    :

`MethodWrapper(func: Any)`
:   Wrapper that adapts a method function to support .apply(model, config).

    ### Methods

    `apply(self, model: Any, config: Any) ‑> Any`
    :