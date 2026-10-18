"""
Distributed REAP MoE Router Utilities

Distributed routing and calibration utilities for robust expert allocation and pruning attention modules.

Author: Taha Majlesi
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from neuralops.attentions.sparse.reap_moe import REAPMoE, REAPMoEAttention

logger = logging.getLogger(__name__)


def calibrate_expert_activations(
    model: nn.Module,
    dataloader_or_dataset: Any,
    world_size: int = 1,
    rank: int = 0,
    max_steps: Optional[int] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Calibrate expert routing probabilities across all ranks with torch.distributed.all_reduce
    if distributed is initialized, else locally.

    Returns a dict mapping module name to its 1D tensor of global expert importance scores.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    is_dist = dist.is_available() and dist.is_initialized()
    if is_dist:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    target_modules: Dict[str, nn.Module] = {}
    for name, module in model.named_modules():
        if isinstance(module, REAPMoE):
            target_modules[name] = module
        elif isinstance(module, REAPMoEAttention) and hasattr(module, "moe"):
            target_modules[name] = module.moe
        elif hasattr(module, "router") and hasattr(module, "experts"):
            target_modules[name] = module

    if not target_modules:
        logger.warning("No REAP MoE modules found in model for calibration.")
        return {}

    importance_accumulators: Dict[str, torch.Tensor] = {}
    token_counts: Dict[str, int] = {}
    for name, module in target_modules.items():
        n_experts = getattr(module, "n_experts", len(getattr(module, "experts", [])))
        importance_accumulators[name] = torch.zeros(n_experts, dtype=torch.float64, device=device)
        token_counts[name] = 0

    model.eval()

    handles = []
    current_activations: Dict[str, torch.Tensor] = {}

    def make_hook(mod_name: str):
        def hook(mod, inputs, outputs):
            with torch.no_grad():
                x = inputs[0]
                if hasattr(mod, "router_ln"):
                    normed_x = mod.router_ln(x)
                else:
                    normed_x = x
                logits = mod.router(normed_x)
                probs = F.softmax(logits.float(), dim=-1).view(-1, logits.shape[-1])
                top_k = getattr(mod, "top_k", 1)
                topk_vals, topk_indices = torch.topk(probs, top_k, dim=-1)
                mask = torch.zeros_like(probs).scatter_(-1, topk_indices, 1.0)
                weighted_mass = (probs * mask).sum(dim=0).to(torch.float64)
                importance_accumulators[mod_name] += weighted_mass
                token_counts[mod_name] += probs.shape[0]
        return hook

    for name, mod in target_modules.items():
        handles.append(mod.register_forward_hook(make_hook(name)))

    try:
        step = 0
        with torch.no_grad():
            for batch in dataloader_or_dataset:
                if isinstance(batch, (tuple, list)):
                    inp = batch[0].to(device)
                elif isinstance(batch, dict):
                    inp = batch.get("input_ids", batch.get("inputs", next(iter(batch.values())))).to(device)
                elif isinstance(batch, torch.Tensor):
                    inp = batch.to(device)
                else:
                    inp = batch

                if hasattr(model, "__call__"):
                    if isinstance(inp, torch.Tensor):
                        model(inp)
                    elif isinstance(batch, dict):
                        model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})

                step += 1
                if max_steps is not None and step >= max_steps:
                    break
    finally:
        for h in handles:
            h.remove()

    global_importance: Dict[str, torch.Tensor] = {}
    for name, accum in importance_accumulators.items():
        total_tokens = torch.tensor(token_counts[name], dtype=torch.float64, device=device)
        if is_dist:
            dist.all_reduce(accum, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        if total_tokens.item() > 0:
            global_importance[name] = (accum / total_tokens).to(torch.float32)
        else:
            global_importance[name] = accum.to(torch.float32)

    return global_importance


def prune_moe_experts(
    moe_module: nn.Module,
    expert_scores: torch.Tensor,
    retain_experts: int,
) -> Tuple[List[int], List[int]]:
    """
    Prune bottom experts based on importance scores and re-normalize remaining routing weights.

    Args:
        moe_module: REAPMoE instance.
        expert_scores: 1D tensor of expert importance scores.
        retain_experts: number of top experts to keep (K).

    Returns:
        (retained_indices, pruned_indices)
    """
    n_experts = getattr(moe_module, "n_experts", len(moe_module.experts))
    retain_experts = max(1, min(retain_experts, n_experts))

    sorted_indices = torch.argsort(expert_scores, descending=True).tolist()
    retained_indices = sorted(sorted_indices[:retain_experts])
    pruned_indices = sorted(sorted_indices[retain_experts:])

    logger.info(
        "REAP Pruning: retaining %d/%d experts: %s (pruned: %s)",
        retain_experts,
        n_experts,
        retained_indices,
        pruned_indices,
    )

    new_experts = nn.ModuleList([moe_module.experts[idx] for idx in retained_indices])
    moe_module.experts = new_experts

    with torch.no_grad():
        retained_tensor = torch.tensor(retained_indices, device=moe_module.router.weight.device)
        if hasattr(moe_module.router, "weight") and moe_module.router.weight is not None:
            new_weight = moe_module.router.weight[retained_tensor, :].clone()
            norm_factor = (n_experts / retain_experts) ** 0.5
            new_weight.mul_(norm_factor)
            moe_module.router.weight = nn.Parameter(new_weight)
            moe_module.router.out_features = retain_experts

        if hasattr(moe_module.router, "bias") and moe_module.router.bias is not None:
            new_bias = moe_module.router.bias[retained_tensor].clone()
            moe_module.router.bias = nn.Parameter(new_bias)

    moe_module.n_experts = retain_experts
    moe_module.top_k = min(getattr(moe_module, "top_k", 1), retain_experts)

    return retained_indices, pruned_indices


def distributed_reap_prune(
    model: nn.Module,
    tokenizer: Any = None,
    dataset: Any = None,
    world_size: int = 1,
    rank: int = 0,
    retain_ratio: float = 0.5,
    retain_experts: Optional[int] = None,
    max_calibration_steps: Optional[int] = 20,
    device: Optional[Union[torch.device, str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Calibrate expert routing probabilities across all ranks with
    torch.distributed.all_reduce(..., op=torch.distributed.ReduceOp.SUM) if distributed initialized,
    else local.
    Prune bottom experts and re-normalize remaining routing weights.

    Args:
        model: Model containing REAPMoE blocks.
        tokenizer: Optional tokenizer for tokenizing text dataset.
        dataset: Calibration dataset or dataloader.
        world_size: Total distributed ranks (auto-detected if distributed initialized).
        rank: Current rank (auto-detected if distributed initialized).
        retain_ratio: Ratio of experts to retain (default: 0.5).
        retain_experts: Explicit number of experts to retain per module.
        max_calibration_steps: Max calibration batches.
        device: Device to place calibration tensors on.

    Returns:
        Summary dict containing retained_indices, pruned_indices, and importance_scores.
    """
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

    logger.info("Starting distributed REAP pruning calibration (rank %d/%d)...", rank, world_size)

    if dataset is None:
        logger.info("No calibration dataset provided. Using synthetic dummy activations.")
        d_model = 768
        for p in model.parameters():
            if p.ndim == 2:
                d_model = p.shape[-1]
                break
        dev = device or next(model.parameters()).device
        dataset = [torch.randn(4, 32, d_model, device=dev) for _ in range(max_calibration_steps or 10)]

    importance_map = calibrate_expert_activations(
        model=model,
        dataloader_or_dataset=dataset,
        world_size=world_size,
        rank=rank,
        max_steps=max_calibration_steps,
        device=device,
    )

    results = {}
    for name, module in model.named_modules():
        target_mod = None
        if isinstance(module, REAPMoE):
            target_mod = module
        elif isinstance(module, REAPMoEAttention) and hasattr(module, "moe"):
            target_mod = module.moe

        if target_mod is not None and name in importance_map:
            scores = importance_map[name]
            curr_n = target_mod.n_experts
            k_retain = (
                retain_experts
                if retain_experts is not None
                else max(1, int(round(curr_n * retain_ratio)))
            )
            retained, pruned = prune_moe_experts(target_mod, scores, k_retain)
            results[name] = {
                "retained_experts": retained,
                "pruned_experts": pruned,
                "importance_scores": scores.tolist(),
                "initial_experts": curr_n,
                "final_experts": k_retain,
            }

    logger.info("Distributed REAP pruning completed successfully across %d modules.", len(results))
    return results


__all__ = [
    "calibrate_expert_activations",
    "prune_moe_experts",
    "distributed_reap_prune",
]
