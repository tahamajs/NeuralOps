"""Hybrid attention and compression layer specification normalizer.

Allows configuring per-layer attention mechanisms or compression engines via
compact dictionaries or lists of dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field


__all__ = ["HybridAttentionSpec", "normalize_hybrid_spec", "get_layer_spec"]


@dataclass
class HybridAttentionSpec:
    """Specification for heterogeneous per-layer attention mechanisms."""

    layer_map: dict[int, str] = field(default_factory=dict)
    default_attention: str = "full_attention"

    def to_list(self, num_layers: int) -> list[dict[str, Any]]:
        specs = []
        for i in range(num_layers):
            attn_type = self.layer_map.get(i, self.default_attention)
            specs.append({"layers": [i], "type": attn_type})
        return specs


def normalize_hybrid_spec(
    spec: Any,
    num_layers: int,
    default_type: str = "svd_linear_attention",
    available_registry: Any = None,
) -> list[dict[str, Any]]:
    """Normalize a hybrid layer specification into a per-layer list of dictionaries.

    Args:
        spec: None, a dict, or a list of dicts specifying layer assignments.
        num_layers: Total number of transformer layers.
        default_type: Fallback module type for unassigned layers.
        available_registry: Optional registry or collection to validate module types.

    Returns:
        List[Dict[str, Any]]: Length num_layers list containing the exact spec for each layer.

    Raises:
        ValueError: If layer index is out of bounds, duplicated, or module is unknown.
    """
    if num_layers <= 0:
        raise ValueError(f"num_layers must be positive, got {num_layers}")

    result: list[dict[str, Any]] = [
        {"type": default_type} for _ in range(num_layers)
    ]

    if spec is None:
        return result

    if isinstance(spec, str):
        # Uniform single type across all layers
        return [{"type": spec} for _ in range(num_layers)]

    if isinstance(spec, HybridAttentionSpec):
        specs_list = spec.to_list(num_layers)
    elif isinstance(spec, dict):
        if any(isinstance(k, int) for k in spec.keys()):
            specs_list = [{"layers": [k], "type": v} for k, v in spec.items()]
        else:
            specs_list = [spec]
    elif isinstance(spec, (list, tuple)):
        specs_list = list(spec)
    else:
        raise ValueError(f"Unsupported spec format: {type(spec)}. Expected dict or list of dicts.")

    assigned_layers: set[int] = set()

    for item in specs_list:
        if not isinstance(item, dict):
            raise ValueError(f"Each spec entry must be a dict, got {type(item)}")

        mod_type = item.get("type", default_type)
        if available_registry is not None and mod_type not in available_registry:
            raise ValueError(f"Module '{mod_type}' is not registered in registry.")

        layers = item.get("layers", None)
        if layers is None:
            # Applies to all unassigned layers
            target_indices = [i for i in range(num_layers) if i not in assigned_layers]
        elif isinstance(layers, int):
            target_indices = [layers]
        elif isinstance(layers, (list, tuple)):
            target_indices = list(layers)
        else:
            raise ValueError(f"Invalid 'layers' field: {layers}")

        for idx in target_indices:
            if idx < 0 or idx >= num_layers:
                raise ValueError(f"Layer index {idx} out of range [0, {num_layers})")
            if idx in assigned_layers:
                raise ValueError(f"Layer {idx} specified more than once in hybrid spec")
            assigned_layers.add(idx)

            layer_kwargs = {k: v for k, v in item.items() if k not in ("layers", "type")}
            result[idx] = {"type": mod_type, **layer_kwargs}

    return result


def get_layer_spec(
    normalized_specs: list[dict[str, Any]],
    layer_idx: int,
) -> dict[str, Any]:
    """Retrieve the specification dictionary for a specific layer.

    Args:
        normalized_specs: The list returned by `normalize_hybrid_spec`.
        layer_idx: Index of the target layer.

    Returns:
        dict[str, Any]: Layer specification parameters.
    """
    if layer_idx < 0 or layer_idx >= len(normalized_specs):
        raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(normalized_specs)})")
    return normalized_specs[layer_idx]
