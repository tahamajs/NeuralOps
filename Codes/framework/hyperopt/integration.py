"""Integration with Django runner and ResultsSaver."""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
from .config import HPOConfig, HPOBackend
from .base import HPO_REGISTRY

def run_hpo_experiment(
    config_dict: Dict[str, Any],
    objective_fn: Callable[[Dict[str, Any]], float],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run an HPO experiment from configuration.

    Args:
        config_dict: dictionary following HPOConfig schema
        objective_fn: function(params) -> float (score)
        output_dir: directory to save results (optional)

    Returns:
        dict with best_params, best_score, and full trial history
    """
    cfg = HPOConfig(**{k: v for k, v in config_dict.items() if k in HPOConfig.__dataclass_fields__})
    optimizer_cls = HPO_REGISTRY.resolve(cfg.backend.value)
    if optimizer_cls is None:
        raise ValueError(f"Unknown HPO backend: {cfg.backend}")
    optimizer = optimizer_cls(cfg) if callable(optimizer_cls) else optimizer_cls(cfg)
    best_params, best_score = optimizer.optimize(objective_fn)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        import json
        with open(output_dir / "hpo_results.json", 'w') as f:
            json.dump({"best_params": best_params, "best_score": best_score}, f, indent=2)

    return {"best_params": best_params, "best_score": best_score}