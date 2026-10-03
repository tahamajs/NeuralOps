from typing import Any


class MethodWrapper:
    """Wrapper that adapts a method function to support .apply(model, config)."""

    def __init__(self, func: Any) -> None:
        self.func = func

    def apply(self, model: Any, config: Any) -> Any:
        if isinstance(config, dict):

            class ConfigObj:
                def __init__(self, d):
                    for k, v in d.items():
                        setattr(self, k, v)

                def __getattr__(self, item):
                    return None

            cfg_obj = ConfigObj(config)
        else:
            cfg_obj = config
        res = self.func(model, cfg_obj)
        return res if res is not None else model


class MethodRegistryDict(dict):
    """Custom dictionary registry with resolve() method for training methods."""

    def resolve(self, name: str) -> Any:
        fn = self.get(name)
        if fn is None:
            return None
        return MethodWrapper(fn)


METHOD_REGISTRY = MethodRegistryDict()


def register_method(name: str) -> Any:
    """Decorator to register a method callable."""

    def decorator(fn) -> Any:
        """Decorator."""
        METHOD_REGISTRY[name] = fn
        return fn

    return decorator


def get_method(name: str) -> Any:
    """Return the method function by name, or None if not found."""
    return METHOD_REGISTRY.get(name)


def available_methods() -> Any:
    """Return list of all registered method names."""
    return list(METHOD_REGISTRY.keys())
