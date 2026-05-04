# framework/trainer/methods/registry.py
METHOD_REGISTRY = {}

def register_method(name):
    """Decorator to register a method (e.g., @register_method('lora'))."""
    def decorator(fn):
        METHOD_REGISTRY[name] = fn
        return fn
    return decorator

def get_method(name):
    """Return the method function by name."""
    return METHOD_REGISTRY.get(name)

def available_methods():
    """Return list of registered method names."""
    return list(METHOD_REGISTRY.keys())