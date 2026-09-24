#!/usr/bin/env python3
"""Offline zero-shot RTE sweep for decoder-only Hugging Face models.

This runner deliberately does not instantiate a random sequence-classification
head.  It evaluates binary RTE by comparing the causal-LM probability of the
answers ``Yes`` and ``No`` after an entailment prompt.  Each registered module
is evaluated on a pristine model instance so a failed injection cannot affect
later results.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

WORKDIR = Path(__file__).resolve().parent
PROJECT_DIR = WORKDIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "Codes"))

from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, is_helper_module
from framework.utils.qwen_compat import install_qwen_aliases, is_architecture_error, is_dtype_error
import framework.attentions  # noqa: F401
import framework.compression_engine  # noqa: F401
from framework.eval_all_models_datasets import apply_module_to_model


LOGGER = logging.getLogger("QwenZeroShotSweep")
CSV_FIELDS = [
    "run_id", "module_name", "model_family", "dataset", "status", "accuracy_pct",
    "examples", "elapsed_s", "peak_vram_mb", "error", "timestamp",
]


def available_modules() -> list[str]:
    """Return a stable canonical module list, excluding registry helpers."""
    # Registry population is module-import driven; importing only the package
    # leaves most optional attention/compressor implementations undiscovered.
    for package in ("framework.attentions", "framework.compression_engine"):
        package_dir = WORKDIR.parent / ("attentions" if package.endswith("attentions") else "compression_engine")
        for source in package_dir.glob("*.py"):
            if source.stem.startswith("__"):
                continue
            try:
                importlib.import_module(f"{package}.{source.stem}")
            except Exception:
                # A module can still be recorded as unavailable during the
                # sweep; one optional import must not hide all other modules.
                continue
    by_factory: dict[object, list[str]] = {}
    for name, factory in list(ATTENTION_REGISTRY.items()) + list(COMPRESSOR_REGISTRY.items()):
        if name == "full" or is_helper_module(name):
            continue
        by_factory.setdefault(factory, []).append(name)
    names = ["full"]
    for aliases in by_factory.values():
        names.append(sorted(aliases, key=lambda item: (item.endswith("_inject"), len(item), item))[0])
    return sorted(set(names), key=lambda item: (item != "full", item))


def prompt_for(example: dict[str, object]) -> str:
    return (
        "Premise: " + str(example["sentence1"]) + "\n"
        "Hypothesis: " + str(example["sentence2"]) + "\n"
        "Does the premise entail the hypothesis? Answer:"
    )


def is_skip_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    fingerprints = (
        "out of memory", "cuda error", "size mismatch", "shape", "unexpected keyword",
        "got an unexpected", "has no attribute", "not enough values to unpack",
        "not implemented", "does not support", "attention mask",
    )
    return any(part in text for part in fingerprints)


@torch.inference_mode()
def evaluate_rte(model, tokenizer, examples: list[dict[str, object]], batch_size: int, seq_len: int, device: torch.device) -> float:
    yes_ids = tokenizer(" Yes", add_special_tokens=False).input_ids
    no_ids = tokenizer(" No", add_special_tokens=False).input_ids
    if not yes_ids or not no_ids:
        raise RuntimeError("Tokenizer did not produce label token IDs for Yes/No")
    yes_id, no_id = yes_ids[0], no_ids[0]
    predictions: list[int] = []
    labels: list[int] = []
    tokenizer.padding_side = "left"
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset : offset + batch_size]
        encoded = tokenizer(
            [prompt_for(item) for item in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_len,
        ).to(device)
        outputs = model(**encoded, use_cache=False)
        next_logits = outputs.logits[:, -1, :]
        # RTE uses 0=entailment and 1=not_entailment.
        predictions.extend((next_logits[:, no_id] > next_logits[:, yes_id]).long().cpu().tolist())
        labels.extend(int(item["label"]) for item in batch)
    return 100.0 * sum(p == y for p, y in zip(predictions, labels)) / max(1, len(labels))


def load_model(model_id: str, device: torch.device, dtype: torch.dtype | None = None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eff_dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            dtype=eff_dtype,
        ).to(device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            local_files_only=True,
            torch_dtype=eff_dtype,
        ).to(device)
    model.eval()
    install_qwen_aliases(model)
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset_dir", default=str(WORKDIR / "data" / "glue_data" / "rte"))
    parser.add_argument("--modules", default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=120)
    # Accepted for parity with the standard sweep CLI. Zero-shot evaluation
    # has no optimizer steps, so this value is intentionally ignored.
    parser.add_argument("--max_steps", type=int, default=1)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "summary.csv"
    errors_path = output_dir / "module_errors.jsonl"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This sweep requires CUDA")
    dataset = load_from_disk(args.dataset_dir)["validation"]
    examples = [dataset[index] for index in range(len(dataset))]
    modules = available_modules() if args.modules == "all" else [item.strip() for item in args.modules.split(",") if item.strip()]
    LOGGER.info("Discovered %d modules for sweep", len(modules))
    run_id = f"qwen_rte_zero_shot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    completed_modules: set[str] = set()
    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as existing:
            for prior in csv.DictReader(existing):
                if prior.get("status") in {"PASS", "SKIP", "FAIL"}:
                    completed_modules.add(prior.get("module_name", ""))
    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with results_path.open("a", newline="", encoding="utf-8") as csv_file, errors_path.open("a", encoding="utf-8") as error_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for index, module_name in enumerate(modules, start=1):
            if module_name in completed_modules:
                LOGGER.info("[%d/%d] %s already recorded; resuming past it", index, len(modules), module_name)
                continue
            started = time.monotonic()
            status, accuracy, error = "PASS", 0.0, ""
            peak_vram = 0.0
            model = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                model, tokenizer = load_model(args.model, device, torch.float16 if args.precision == "fp16" else torch.float32)
                if module_name != "full":
                    module_type = "attention" if module_name in ATTENTION_REGISTRY else "compressor"
                    if module_name not in ATTENTION_REGISTRY and module_name not in COMPRESSOR_REGISTRY:
                        raise RuntimeError(f"Module is not registered: {module_name}")
                    model, _ = apply_module_to_model(model, module_name, module_type, rank=16, device=device)
                    model.eval()
                accuracy = round(evaluate_rte(model, tokenizer, examples, args.batch_size, args.seq_len, device), 2)
                peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
            except Exception as exc:  # Keep an individual module from stopping the sweep.
                # Retry decomposition-heavy modules once with a full fp32
                # model. This also catches CUDA QR/SVD kernels unavailable in
                # half precision without changing the normal fp16 sweep.
                if module_name != "full" and is_dtype_error(exc):
                    try:
                        del model
                        gc.collect()
                        torch.cuda.empty_cache()
                        model, tokenizer = load_model(args.model, device, torch.float32)
                        module_type = "attention" if module_name in ATTENTION_REGISTRY else "compressor"
                        model, _ = apply_module_to_model(model, module_name, module_type, rank=16, device=device)
                        model.eval()
                        accuracy = round(evaluate_rte(model, tokenizer, examples, args.batch_size, args.seq_len, device), 2)
                        peak_vram = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
                        error = f"fp32 retry after {type(exc).__name__}: {exc}"
                        exc = None
                    except Exception as retry_exc:
                        exc = retry_exc
                if exc is not None:
                    status = "SKIP" if is_architecture_error(exc) or is_skip_error(exc) else "FAIL"
                    error = f"{type(exc).__name__}: {exc}"
                    error_file.write(json.dumps({"module": module_name, "status": status, "error": error}) + "\n")
                    LOGGER.warning("[%d/%d] %s -> %s: %s", index, len(modules), module_name, status, error)
            finally:
                if model is not None:
                    del model
                gc.collect()
                torch.cuda.empty_cache()
            row = {
                "run_id": run_id, "module_name": module_name, "model_family": "QWEN", "dataset": "RTE",
                "status": status, "accuracy_pct": accuracy, "examples": len(examples),
                "elapsed_s": round(time.monotonic() - started, 2), "peak_vram_mb": peak_vram,
                "error": error[:1000], "timestamp": datetime.now().isoformat(),
            }
            writer.writerow(row)
            csv_file.flush()
            LOGGER.info("[%d/%d] %s -> %s, accuracy=%.2f%%", index, len(modules), module_name, status, accuracy)


if __name__ == "__main__":
    main()
