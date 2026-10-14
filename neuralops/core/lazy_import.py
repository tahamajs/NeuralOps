"""
PEP 562 Lazy Module Loader and dynamic proxy for high-speed framework startup.
Prevents heavy CUDA / OpenMP / TorchInductor eager initialization until symbols are accessed.
"""

from __future__ import annotations

import importlib
import types
from typing import Any, Dict, List, Optional, Set, Tuple


class _LazyModule(types.ModuleType):
    """
    Lazy module implementation following PEP 562 and HuggingFace transformers conventions.
    Defers importing heavy submodules or attributes until attribute access.
    """

    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: Dict[str, List[str]],
        module_spec: Any = None,
        extra_objects: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(name)
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [module_file.rsplit("/", 1)[0]] if "/" in module_file else []
        self._import_structure = import_structure
        self._reverse_map: Dict[str, str] = {}
        for mod, symbols in import_structure.items():
            for symbol in symbols:
                self._reverse_map[symbol] = mod

        self._extra_objects = extra_objects or {}
        for k, v in self._extra_objects.items():
            setattr(self, k, v)

        self._loaded_submodules: Set[str] = set()

    def __getattr__(self, name: str) -> Any:
        if name in self._extra_objects:
            return self._extra_objects[name]

        if name in self._reverse_map:
            submodule_name = self._reverse_map[name]
            submod = self._load_submodule(submodule_name)
            val = getattr(submod, name)
            setattr(self, name, val)
            return val

        if name in self._import_structure:
            submod = self._load_submodule(name)
            setattr(self, name, submod)
            return submod

        raise AttributeError(f"module '{self.__name__}' has no attribute '{name}'")

    def _load_submodule(self, submodule_name: str) -> types.ModuleType:
        if submodule_name.startswith("."):
            full_name = f"{self.__name__}{submodule_name}"
        elif "." in submodule_name:
            full_name = submodule_name
        else:
            full_name = f"{self.__name__}.{submodule_name}"

        mod = importlib.import_module(full_name)
        self._loaded_submodules.add(submodule_name)
        return mod

    def __dir__(self) -> List[str]:
        res = set(super().__dir__())
        res.update(self._reverse_map.keys())
        res.update(self._import_structure.keys())
        res.update(self._extra_objects.keys())
        return sorted(list(res))
