# framework/core/registry.py
"""
Plugin registry system for framework components.

This module provides a simple registry pattern to register and retrieve
components like attention mechanisms, models, and visualizers.
"""

from typing import Dict, Any, Optional, Callable


class Registry:
    """
    A simple registry for plugins (models, attentions, visualizers).
    
    The registry allows components to be registered with a key and
    retrieved later by that key.
    
    Example:
        >>> registry = Registry("attentions")
        >>> 
        >>> @registry.register("my_attention")
        ... class MyAttention:
        ...     pass
        >>> 
        >>> attn_class = registry.get("my_attention")
    """
    
    def __init__(self, name: str):
        """
        Args:
            name: Name of the registry (e.g., "models", "attentions")
        """
        self.name = name
        self._dict: Dict[str, Callable] = {}
    
    def register(self, key: Optional[str] = None) -> Callable:
        """
        Decorator to register a class or function.
        
        Args:
            key: Optional custom key. If None, uses the class/function name.
            
        Returns:
            Decorator function
        
        Example:
            >>> @registry.register()
            ... class MyClass:
            ...     pass
            
            >>> @registry.register("custom_key")
            ... class AnotherClass:
            ...     pass
        """
        def decorator(obj: Callable) -> Callable:
            nonlocal key
            if key is None:
                key = obj.__name__
            if key in self._dict:
                import warnings
                warnings.warn(
                    f"Registry '{self.name}' already contains key '{key}'. "
                    f"Overwriting with {obj}.",
                    UserWarning
                )
            self._dict[key] = obj
            return obj
        return decorator
    
    def get(self, key: str) -> Optional[Callable]:
        """
        Retrieve a registered component by key.
        
        Args:
            key: The registration key
            
        Returns:
            The registered class/function, or None if not found
        """
        return self._dict.get(key)
    
    def keys(self) -> Iterable[str]:
        """Return all registered keys."""
        return self._dict.keys()
    
    def values(self) -> Iterable[Callable]:
        """Return all registered values."""
        return self._dict.values()
    
    def items(self) -> Iterable[tuple]:
        """Return all key-value pairs."""
        return self._dict.items()
    
    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the registry."""
        return key in self._dict
    
    def __len__(self) -> int:
        """Number of registered items."""
        return len(self._dict)
    
    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', items={len(self._dict)})"


# ----------------------------------------------------------------------------
# Global Registries
# ----------------------------------------------------------------------------

MODEL_REGISTRY = Registry("models")
"""
Registry for model classes (e.g., BertForSequenceClassification).
Usage:
    @MODEL_REGISTRY.register("bert-base-uncased")
    class CustomBert(BertForSequenceClassification):
        pass
"""

ATTENTION_REGISTRY = Registry("attentions")
"""
Registry for attention mechanisms.
Usage:
    @ATTENTION_REGISTRY.register("svd")
    class SVDLinearAttention(BaseAttention):
        pass
"""

VISUALIZER_REGISTRY = Registry("visualizers")
"""
Registry for visualizers.
Usage:
    @VISUALIZER_REGISTRY.register("attention_heatmap")
    class AttentionHeatmapVisualizer(BaseVisualizer):
        pass
"""

# ----------------------------------------------------------------------------
# Helper Functions
# ----------------------------------------------------------------------------

def list_registered_components(registry: Registry) -> Dict[str, str]:
    """
    List all registered components in a registry.
    
    Args:
        registry: The registry to inspect
        
    Returns:
        Dictionary mapping keys to class names
    """
    return {key: value.__name__ for key, value in registry.items()}


def print_registry_summary():
    """Print a summary of all registered components."""
    print("\n" + "=" * 60)
    print("📦 Registered Components")
    print("=" * 60)
    
    registries = {
        "Models": MODEL_REGISTRY,
        "Attentions": ATTENTION_REGISTRY,
        "Visualizers": VISUALIZER_REGISTRY,
    }
    
    for name, registry in registries.items():
        print(f"\n🔹 {name}:")
        if len(registry) == 0:
            print("   (none registered yet)")
        else:
            for key in registry.keys():
                print(f"   - {key}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Example usage
    print_registry_summary()