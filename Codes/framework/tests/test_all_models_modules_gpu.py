"""
Comprehensive GPU Multi-Model & Multi-Module Verification Matrix Test Suite.

Systematically verifies every attention module in ATTENTION_REGISTRY and every
compressor in COMPRESSOR_REGISTRY across all available model backbones
(Qwen3.5, GPT-2, RoBERTa, DistilBERT) on GPU (`cuda`).

Tests for each (model, module) pair:
1. Architecture compatibility inspection.
2. Safe module injection / compression into the backbone.
3. GPU forward pass with token inputs & attention masks.
4. Numerical sanity check (no NaN, no Inf).
5. Autograd backward pass & gradient propagation for trainable parameters.
6. Peak VRAM tracking and automatic GPU memory cleanup.
7. Structured summary table and CSV / JSON export.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Ensure framework is in path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Framework core imports
from framework.core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    is_module_declared_compatible,
)
from framework.core.attention_utils import inject_attention
from framework.utils.compat import module_family_support, wrap_for_family
from framework.utils.qwen_compat import install_qwen_aliases, is_qwen_model
from framework.utils.model_family import detect_family, ModelAdapter
from framework.eval_all_models_datasets import get_dataset_batch, _extract_logits_and_loss

# Ensure attention and compression modules are fully registered
import framework.attentions
import framework.compression_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GPUModuleModelMatrixTest")

# Available models configuration dictionary
DEFAULT_MODEL_PATHS: Dict[str, Dict[str, Any]] = {
    "qwen3_5": {
        "family": "qwen3_5",
        "local_paths": [
            "PROJECT_ROOT/models/Qwen3.5-0.8B",
            "models/Qwen3.5-0.8B",
            "Qwen/Qwen2.5-0.5B",
        ],
        "hf_fallback": "Qwen/Qwen2.5-0.5B",
        "seq_len": 32,
        "batch_size": 2,
    },
    "gpt2": {
        "family": "gpt2",
        "local_paths": [
            "PROJECT_ROOT/models/gpt2",
            "models/gpt2",
            "gpt2",
        ],
        "hf_fallback": "gpt2",
        "seq_len": 32,
        "batch_size": 2,
    },
    "roberta": {
        "family": "roberta",
        "local_paths": [
            "PROJECT_ROOT/models/roberta-base",
            "models/roberta-base",
            "roberta-base",
        ],
        "hf_fallback": "roberta-base",
        "seq_len": 32,
        "batch_size": 2,
    },
    "distilbert": {
        "family": "distilbert",
        "local_paths": [
            "PROJECT_ROOT/models/distilbert_sst2",
            "models/distilbert_sst2",
            "distilbert-base-uncased",
        ],
        "hf_fallback": "distilbert-base-uncased",
        "seq_len": 32,
        "batch_size": 2,
    },
}


def resolve_model_path(model_key: str) -> Tuple[str, str]:
    """Resolve the best existing local directory or fallback for a model key."""
    cfg = DEFAULT_MODEL_PATHS.get(model_key.lower())
    if not cfg:
        return model_key, model_key
    for p in cfg["local_paths"]:
        if Path(p).exists():
            return p, cfg["family"]
    return cfg["hf_fallback"], cfg["family"]


def load_backbone(
    model_path: str,
    family: str,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, Any]:
    """Load a fresh backbone model on the target device."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if dtype is None:
        dtype = torch.float16 if (device.type == "cuda" and family in {"qwen", "qwen3_5", "qwen2"}) else torch.float32

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            dtype=dtype,
            local_files_only=Path(model_path).exists(),
        ).to(device)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            torch_dtype=dtype,
            local_files_only=Path(model_path).exists(),
        ).to(device)

    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = 0

    # Apply family-specific patches / aliases
    if family in {"qwen", "qwen3_5", "qwen2"}:
        install_qwen_aliases(model)
    wrap_for_family(model, family)

    return model, model.config


def run_single_module_test(
    model: nn.Module,
    module_name: str,
    family: str,
    device: torch.device,
    seq_len: int = 32,
    batch_size: int = 2,
    test_backward: bool = True,
) -> Dict[str, Any]:
    """
    Test a single module (attention or compressor) on a clone of the model.
    """
    result: Dict[str, Any] = {
        "module": module_name,
        "family": family,
        "status": "PASS",
        "type": "unknown",
        "elapsed_s": 0.0,
        "peak_vram_mb": 0.0,
        "forward_finite": False,
        "backward_finite": False,
        "error": "",
    }

    t0 = time.monotonic()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # 1. Compatibility Check
    is_compat, reason = is_module_declared_compatible(module_name, family)
    support = module_family_support(module_name).get(family, "unknown")

    if not is_compat:
        result["status"] = f"SKIP_{family.upper()}_INCOMPATIBLE_DECLARED"
        result["error"] = reason
        result["elapsed_s"] = round(time.monotonic() - t0, 3)
        return result

    if support == "incompatible":
        result["status"] = "SKIP_NO_SUPPORT"
        result["error"] = "unsupported_family"
        result["elapsed_s"] = round(time.monotonic() - t0, 3)
        return result

    # 2. Module Type Determination
    is_attn = module_name in ATTENTION_REGISTRY
    is_comp = module_name in COMPRESSOR_REGISTRY
    if not is_attn and not is_comp:
        result["status"] = "ERROR_NOT_FOUND"
        result["error"] = f"Module '{module_name}' not registered in ATTENTION_REGISTRY or COMPRESSOR_REGISTRY"
        result["elapsed_s"] = round(time.monotonic() - t0, 3)
        return result

    result["type"] = "attention" if is_attn else "compressor"

    # Deepcopy model to ensure complete isolation
    try:
        candidate = copy.deepcopy(model).to(device)
    except Exception as e:
        result["status"] = "FAIL_CLONE"
        result["error"] = f"deepcopy failed: {str(e)}"
        result["elapsed_s"] = round(time.monotonic() - t0, 3)
        return result

    # 3. Injection / Compression Step
    try:
        from framework.eval_all_models_datasets import apply_module_to_model
        candidate, _ = apply_module_to_model(candidate, module_name, result["type"], rank=16, device=device)
    except Exception as e:
        result["status"] = "FAIL_INJECTION"
        result["error"] = f"injection error: {str(e)[:300]}"
        result["elapsed_s"] = round(time.monotonic() - t0, 3)
        del candidate
        return result

    # 4. Forward Pass Check
    try:
        candidate.train()
        batch = get_dataset_batch(family, batch_size=batch_size, seq_len=seq_len, device=device)
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")

        outputs = candidate(input_ids=input_ids, attention_mask=attention_mask)
        logits, loss = _extract_logits_and_loss(outputs)

        if logits is None:
            result["status"] = "FAIL_FORWARD_NO_LOGITS"
            result["error"] = "Model returned None for logits"
            result["elapsed_s"] = round(time.monotonic() - t0, 3)
            del candidate
            return result

        if torch.isnan(logits).any() or torch.isinf(logits).any():
            result["status"] = "FAIL_FORWARD_NAN"
            result["error"] = "Logits contain NaN or Inf values"
            result["elapsed_s"] = round(time.monotonic() - t0, 3)
            del candidate
            return result

        result["forward_finite"] = True

    except Exception as e:
        result["status"] = "FAIL_FORWARD"
        result["error"] = f"forward error: {str(e)[:300]}"
        result["elapsed_s"] = round(time.monotonic() - t0, 3)
        del candidate
        return result

    # 5. Backward Pass Check (Autograd test)
    if test_backward:
        try:
            target_loss = loss if (loss is not None and torch.is_tensor(loss)) else logits.sum()
            candidate.zero_grad(set_to_none=True)
            target_loss.backward()

            # Verify that at least some parameters have finite gradients
            has_grad = False
            grad_nan = False
            for p in candidate.parameters():
                if p.grad is not None:
                    has_grad = True
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        grad_nan = True
                        break

            if grad_nan:
                result["status"] = "FAIL_BACKWARD_NAN"
                result["error"] = "Gradients contain NaN or Inf"
            elif not has_grad:
                result["backward_finite"] = True  # e.g., if all params frozen in pure post-hoc compressor
            else:
                result["backward_finite"] = True

        except Exception as e:
            result["status"] = "FAIL_BACKWARD"
            result["error"] = f"backward error: {str(e)[:300]}"
            result["elapsed_s"] = round(time.monotonic() - t0, 3)
            del candidate
            return result

    # Success and Memory Stats
    result["elapsed_s"] = round(time.monotonic() - t0, 3)
    if device.type == "cuda":
        result["peak_vram_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024**2), 1)

    # Cleanup
    del candidate
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


def run_full_gpu_matrix(
    models_to_test: List[str],
    modules_to_test: List[str],
    output_dir: Path,
    device_str: str = "cuda",
) -> List[Dict[str, Any]]:
    """Execute the full matrix of models and modules on the designated device."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if (device_str == "cuda" and torch.cuda.is_available()) else "cpu")
    logger.info("=" * 80)
    logger.info("LAUNCHING GPU MULTI-MODEL MODULE VERIFICATION MATRIX")
    logger.info("  Device:    %s", device)
    logger.info("  Models:    %s", models_to_test)
    logger.info("  Modules:   %d requested", len(modules_to_test))
    logger.info("  Output:    %s", output_dir)
    logger.info("=" * 80)

    csv_path = output_dir / "matrix_test_summary.csv"
    json_path = output_dir / "matrix_test_summary.json"
    fields = [
        "model",
        "family",
        "module",
        "type",
        "status",
        "forward_finite",
        "backward_finite",
        "elapsed_s",
        "peak_vram_mb",
        "error",
    ]

    results: List[Dict[str, Any]] = []

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for model_key in models_to_test:
            model_path, family = resolve_model_path(model_key)
            logger.info("\n>>> Loading backbone for '%s' (%s) from '%s'...", model_key, family, model_path)

            try:
                base_model, config = load_backbone(model_path, family, device)
            except Exception as e:
                logger.error("Failed to load backbone '%s': %s", model_key, e)
                for mod in modules_to_test:
                    row = {
                        "model": model_key,
                        "family": family,
                        "module": mod,
                        "type": "unknown",
                        "status": "FAIL_LOAD_BACKBONE",
                        "forward_finite": False,
                        "backward_finite": False,
                        "elapsed_s": 0.0,
                        "peak_vram_mb": 0.0,
                        "error": str(e)[:300],
                    }
                    writer.writerow(row)
                    f.flush()
                    results.append(row)
                continue

            for idx, mod in enumerate(modules_to_test, 1):
                logger.info(
                    "[%s] [%d/%d] Testing module: %s",
                    model_key,
                    idx,
                    len(modules_to_test),
                    mod,
                )
                res = run_single_module_test(
                    model=base_model,
                    module_name=mod,
                    family=family,
                    device=device,
                )
                row = {
                    "model": model_key,
                    "family": family,
                    "module": mod,
                    "type": res["type"],
                    "status": res["status"],
                    "forward_finite": res["forward_finite"],
                    "backward_finite": res["backward_finite"],
                    "elapsed_s": res["elapsed_s"],
                    "peak_vram_mb": res["peak_vram_mb"],
                    "error": res["error"],
                }
                writer.writerow(row)
                f.flush()
                results.append(row)

                log_fn = logger.info if res["status"] == "PASS" else logger.warning
                log_fn(
                    "  -> %s (type=%s, elapsed=%.2fs, vram=%.1fMB, err=%s)",
                    res["status"],
                    res["type"],
                    res["elapsed_s"],
                    res["peak_vram_mb"],
                    res["error"] or "none",
                )

            # Clean up base model before loading next
            del base_model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("\n" + "=" * 80)
    logger.info("GPU MATRIX TEST COMPLETE")
    logger.info("Summary saved to: %s and %s", csv_path, json_path)
    logger.info("=" * 80)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Full GPU Matrix Test for Models & Modules")
    parser.add_argument(
        "--models",
        type=str,
        default="qwen3_5,gpt2,roberta,distilbert",
        help="Comma-separated list of model keys to test (qwen3_5, gpt2, roberta, distilbert)",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="all",
        help="Comma-separated module names or 'all'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gpu_matrix_test",
        help="Output directory for test reports",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    if args.modules.strip() == "all":
        all_modules = sorted(list(set(list(ATTENTION_REGISTRY.keys()) + list(COMPRESSOR_REGISTRY.keys()))))
    else:
        all_modules = [m.strip() for m in args.modules.split(",") if m.strip()]

    run_full_gpu_matrix(
        models_to_test=models,
        modules_to_test=all_modules,
        output_dir=Path(args.output_dir),
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
