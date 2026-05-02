"""Method registry – maps string to application function."""

METHOD_REGISTRY = {}

def register_method(name):
    def decorator(fn):
        METHOD_REGISTRY[name] = fn
        return fn
    return decorator

def get_method(name):
    return METHOD_REGISTRY.get(name)

def available_methods():
    return list(METHOD_REGISTRY.keys())