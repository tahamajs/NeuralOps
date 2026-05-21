"""
Model sharding utilities inspired by Petals and Hivemind.

Determines how to split a model across federation nodes based on:
- Number of layers
- Node capacities (VRAM, bandwidth, compute)
- Sharding strategy (layer-wise, tensor parallel, etc.)
"""
import math


def compute_shard_assignment(
    num_layers: int,
    nodes: list,  # list of FederationNode objects
    strategy: str = 'layer_wise'
) -> list:
    """
    Returns a list of (node, start_layer, end_layer) tuples.
    For layer-wise, splits contiguous blocks of layers.
    For data parallel, all nodes get all layers.
    For tensor parallel, splits individual layers (simplified).
    """
    if not nodes:
        return []

    if strategy == 'data_parallel':
        # Every node gets all layers
        return [(n, 0, num_layers) for n in nodes]

    elif strategy == 'layer_wise':
        # Distribute layers proportionally to VRAM
        total_vram = sum(n.listing.vram_gb for n in nodes)
        if total_vram == 0:
            # Equal split
            layers_per_node = math.ceil(num_layers / len(nodes))
            assignments = []
            start = 0
            for n in nodes:
                end = min(start + layers_per_node, num_layers)
                assignments.append((n, start, end))
                start = end
            return assignments

        # Proportional split
        assignments = []
        cum_frac = 0.0
        for n in nodes:
            frac = n.listing.vram_gb / total_vram
            start = int(cum_frac * num_layers)
            cum_frac += frac
            end = int(cum_frac * num_layers) if n != nodes[-1] else num_layers
            assignments.append((n, start, end))
        return assignments

    elif strategy == 'tensor_parallel':
        # For tensor parallel, we split each layer's weight matrices across nodes.
        # This is a simplified model: all nodes must participate equally.
        return [(n, 0, num_layers) for n in nodes]

    else:
        raise ValueError(f"Unknown sharding strategy: {strategy}")