# framework/core/registry.py
"""
Enhanced registry system with auto-discovery and plugin support.
"""

import importlib
import pkgutil
import inspect
from typing import Dict, Any, Callable, Optional, List, Type
from pathlib import Path


class Registry:
    """
    Enhanced registry with auto-discovery and plugin support.
    
    Features:
        - Register components with decorators
        - Auto-discover plugins from directories
        - List all registered components
        - Get component by name with fallback
    
    Example:
        >>> registry = Registry("attentions")
        >>> 
        >>> @registry.register("my_attention")
        ... class MyAttention:
        ...     pass
        >>> 
        >>> # Auto-discover from directory
        >>> registry.auto_discover("framework/attentions/custom")
    """
    
    def __init__(self, name: str):
        self.name = name
        self._dict: Dict[str, Callable] = {}
        self._aliases: Dict[str, str] = {}
    
    def register(self, key: Optional[str] = None, alias: Optional[str] = None) -> Callable:
        """
        Decorator to register a class or function.
        
        Args:
            key: Registration key (defaults to class name)
            alias: Alternative name for backward compatibility
            
        Example:
            @registry.register("svd")
            class SVDAttention:
                pass
        """
        def decorator(obj: Callable) -> Callable:
            nonlocal key
            if key is None:
                key = obj.__name__
            
            self._dict[key.lower()] = obj
            
            if alias:
                self._aliases[alias.lower()] = key.lower()
            
            return obj
        return decorator
    
    def get(self, key: str) -> Optional[Callable]:
        """Get registered component by key."""
        key_lower = key.lower()
        
        # Check direct match
        if key_lower in self._dict:
            return self._dict[key_lower]
        
        # Check aliases
        if key_lower in self._aliases:
            return self._dict[self._aliases[key_lower]]
        
        return None
    
    def get_all(self) -> Dict[str, Callable]:
        """Get all registered components."""
        return self._dict.copy()
    
    def keys(self) -> List[str]:
        """Return all registered keys."""
        return list(self._dict.keys())
    
    def auto_discover(self, package_path: str, prefix: str = ""):
        """
        Auto-discover and register components from directory.
        
        Args:
            package_path: Path to directory containing modules
            prefix: Prefix for registration keys
        """
        path = Path(package_path)
        if not path.exists():
            return
        
        for module_info in pkgutil.iter_modules([str(path)]):
            module_name = module_info.name
            if module_name.startswith('_'):
                continue
            
            try:
                # Import module
                spec = importlib.util.spec_from_file_location(
                    f"{prefix}.{module_name}" if prefix else module_name,
                    path / f"{module_name}.py"
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Register classes that inherit from base classes
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if hasattr(obj, '__base__') and obj.__base__.__name__ in ['BaseAttention', 'BaseCompressor']:
                            reg_key = f"{prefix}_{module_name}" if prefix else module_name
                            self.register(reg_key)(obj)
                            
            except Exception as e:
                print(f"Warning: Failed to load module {module_name}: {e}")
    
    def __contains__(self, key: str) -> bool:
        return key.lower() in self._dict or key.lower() in self._aliases
    
    def __len__(self) -> int:
        return len(self._dict)
    
    def __repr__(self) -> str:
        return f"Registry(name='{self.name}', items={list(self._dict.keys())})"


# Global registries


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

# Global registries
ATTENTION_REGISTRY = Registry("attentions")
COMPRESSOR_REGISTRY = Registry("compressors")
TRAINER_REGISTRY = Registry("trainers")
VISUALIZER_REGISTRY = Registry("visualizers")
DATASET_REGISTRY = Registry("datasets")
OPTIMIZER_REGISTRY = Registry("optimizers")
SCHEDULER_REGISTRY = Registry("schedulers")


def list_all_components() -> Dict[str, List[str]]:
    """List all registered components by type."""
    return {
        "attentions": ATTENTION_REGISTRY.keys(),
        "compressors": COMPRESSOR_REGISTRY.keys(),
        "trainers": TRAINER_REGISTRY.keys(),
        "visualizers": VISUALIZER_REGISTRY.keys(),
        "datasets": DATASET_REGISTRY.keys(),
    }

if __name__ == "__main__":
    # Example usage
    print_registry_summary()