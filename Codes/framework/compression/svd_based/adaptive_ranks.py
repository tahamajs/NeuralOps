def get_adaptive_ranks(model_name: str, patterns: list) -> dict:
    """Returns a dictionary mapping pattern → rank."""
    base_low = 32
    base_high = 64
    ranks = {}
    for pattern in patterns:
        if 'out' in pattern.lower() or 'output' in pattern.lower():
            ranks[pattern] = base_high
        elif 'q_lin' in pattern.lower() or 'k_lin' in pattern.lower() or 'v_lin' in pattern.lower():
            ranks[pattern] = base_low
        else:
            ranks[pattern] = base_high
    return ranks

def apply_adaptive_svd(model, rank_map):
    """Apply SVDLinearLayer to matched layers with given ranks."""
    from .svd_linear import SVDLinearLayer
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            for pattern, rank in rank_map.items():
                if pattern in name.lower():
                    try:
                        new_layer = SVDLinearLayer(module, rank)
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        parent = model.get_submodule(parent_name) if parent_name else model
                        setattr(parent, child_name, new_layer)
                        count += 1
                    except Exception:
                        pass
                    break
    return count