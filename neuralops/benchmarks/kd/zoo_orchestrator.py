#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Multi-Module Zoo Execution Orchestrator for NeuralOps Benchmarks.

Manages:
  - Causal dataset chunking and caching (WikiText-2, etc.)
  - Architecture surgery (Qwen/TinyLlama attention transplants)
  - Multi-key iteration with crash-safe resume via `metrics.jsonl`
  - Surgical delta persistence (`attention_weights.safetensors`, `surgical_manifest.json`)
  - Unified Hugging Face model card generation
  - Metrics recording and optional background upload dispatch
"""
from __future__ import annotations

import datetime
import gc
import inspect
import json
import logging
import math
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from neuralops.core.registry import ATTENTION_REGISTRY, create_attention
from neuralops.hub.model_card import generate_unified_model_card
from neuralops.hub.uploader import AsyncHFHubUploader
from neuralops.layers.qwen_adapter import QwenAttentionAdapter
from neuralops.models.surgical_io import extract_surgical_manifest, save_surgical_delta

logger = logging.getLogger("neuralops.benchmarks.kd.zoo_orchestrator")


class StratifiedQwenDataset(Dataset):
    """Dataset yielding token chunks of fixed length for causal LM training."""

    def __init__(self, token_chunks: torch.Tensor) -> None:
        self.chunks = token_chunks

    def __len__(self) -> int:
        return self.chunks.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.chunks[idx].clone()
        labels = ids.clone()
        attn_mask = torch.ones_like(ids, dtype=torch.long)
        return {
            "input_ids": ids,
            "attention_mask": attn_mask,
            "labels": labels,
        }


CausalTextDataset = StratifiedQwenDataset


def load_tokenized_qwen_corpus(
    tokenizer: Any,
    dataset_name: str = "wikitext-2-raw-v1",
    seq_len: int = 512,
    max_train_samples: int = 4000,
    cache_path_override: Optional[str] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenizes corpus into contiguous chunks of length seq_len with RAM caching."""
    cache_path = Path(cache_path_override) if cache_path_override else Path(f"/dev/shm/wikitext_qwen_seq{seq_len}.pt")
    if cache_path.exists():
        try:
            logger.info("Loading pre-tokenized corpus from RAM cache: %s", cache_path)
            data = torch.load(cache_path, map_location="cpu")
            return data["train_chunks"], data["val_chunks"]
        except Exception as e:
            logger.warning("RAM cache load failed (%s); re-tokenizing...", e)

    from datasets import load_dataset

    logger.info("Loading %s dataset...", dataset_name)
    if "wikitext" in dataset_name:
        raw = load_dataset("wikitext", dataset_name)
        train_texts = [t for t in raw["train"]["text"] if t.strip()]
        val_texts = [t for t in raw["validation"]["text"] if t.strip()]
    else:
        raw = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_texts = [t for t in raw["train"]["text"] if t.strip()]
        val_texts = [t for t in raw["validation"]["text"] if t.strip()]

    def text_to_chunks(texts: list[str], max_chunks: Optional[int] = None) -> torch.Tensor:
        token_parts: list[torch.Tensor] = []
        target_tokens = (max_chunks * seq_len) if max_chunks is not None else None
        collected = 0
        for start in range(0, len(texts), 512):
            batch_txt = "\n\n".join(texts[start : start + 512])
            ids = tokenizer(batch_txt, return_tensors="pt", add_special_tokens=False)["input_ids"].squeeze(0)
            token_parts.append(ids)
            collected += int(ids.numel())
            if target_tokens is not None and collected >= target_tokens:
                break
        tokens = torch.cat(token_parts, dim=0) if token_parts else torch.empty(0, dtype=torch.long)
        n = (tokens.size(0) // seq_len) * seq_len
        chunks = tokens[:n].view(-1, seq_len)
        if max_chunks is not None and chunks.size(0) > max_chunks:
            chunks = chunks[:max_chunks]
        return chunks

    train_chunks = text_to_chunks(train_texts, max_train_samples)
    val_chunks = text_to_chunks(val_texts, 500)

    try:
        torch.save({"train_chunks": train_chunks, "val_chunks": val_chunks}, cache_path)
        logger.info("Cached tokenized corpus to RAM: %s", cache_path)
    except Exception as e:
        logger.debug("Failed to write RAM cache: %s", e)

    logger.info(
        "Corpus tokenized: %d train chunks, %d validation chunks (seq_len=%d)",
        train_chunks.size(0),
        val_chunks.size(0),
        seq_len,
    )
    return train_chunks, val_chunks


def build_cache_wikitext(
    tokenizer: Any,
    seq_len: int = 512,
    cache_path: str = "/dev/shm/wikitext_tinyllama_seq512.pt",
    max_train_samples: int = 4000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build or load cached WikiText-2 token chunks for a tokenizer-specific cache path."""
    return load_tokenized_qwen_corpus(
        tokenizer=tokenizer,
        dataset_name="wikitext-2-raw-v1",
        seq_len=seq_len,
        max_train_samples=max_train_samples,
        cache_path_override=cache_path,
    )


def apply_qwen_attention_surgery(
    model: nn.Module,
    attn_key: str,
    pattern: str = "all",
) -> nn.Module:
    """
    Transplants NeuralOps attention mechanism into Qwen/Llama decoder layers.

    Patterns:
      - 'all': replace all attention layers
      - 'hybrid_3_1': replace 3 subquadratic layers per 1 full quadratic layer
    """
    hidden_size = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_heads)
    head_dim = getattr(model.config, "head_dim", hidden_size // num_heads)

    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise ValueError("Cannot locate transformer layers in model architecture.")

    num_layers = len(layers)
    transplanted_count = 0

    for idx, layer in enumerate(layers):
        if pattern == "hybrid_3_1" and (idx % 4 == 3):
            continue

        orig_attn = getattr(layer, "self_attn", None)
        if orig_attn is None:
            continue

        if attn_key in ATTENTION_REGISTRY:
            try:
                core_attn = create_attention(
                    attn_key,
                    dim=hidden_size,
                    num_heads=num_heads,
                    config=model.config,
                    is_causal=True,
                    layer_idx=idx,
                )
            except TypeError:
                try:
                    core_attn = create_attention(
                        attn_key,
                        dim=hidden_size,
                        num_heads=num_heads,
                        config=model.config,
                        is_causal=True,
                    )
                except TypeError:
                    core_attn = create_attention(
                        attn_key,
                        dim=hidden_size,
                        num_heads=num_heads,
                        config=model.config,
                    )
        else:
            raise KeyError(f"Attention key '{attn_key}' not found in ATTENTION_REGISTRY.")

        adapter = QwenAttentionAdapter(
            core_attention=core_attn,
            embed_dim=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            inherit_weights_from=orig_attn,
            is_causal=True,
            bypass_internal_out_proj=True,
        )

        layer.self_attn = adapter
        transplanted_count += 1

    logger.info(
        "Surgery complete: transplanted %d/%d layers with %s (pattern=%s)",
        transplanted_count,
        num_layers,
        attn_key,
        pattern,
    )
    return model


def normalize_base_model_id(model_name: str) -> str:
    """Normalize local model cache paths to canonical HF Hub IDs."""
    cleaned = str(model_name).strip()
    if "tinyllama" in cleaned.lower():
        return "TinyLlama/TinyLlama_v1.1"
    if "qwen2.5-0.5b" in cleaned.lower():
        return "Qwen/Qwen2.5-0.5B"
    if "llava" in cleaned.lower():
        return "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    if cleaned.startswith("/") or os.path.exists(cleaned):
        tail = Path(cleaned.rstrip("/")).name
        if tail.lower().startswith("qwen"):
            return f"Qwen/{tail}"
        if tail.lower().startswith("tinyllama"):
            return f"TinyLlama/{tail}"
        return tail
    return cleaned


def summarize_attention_doc(attn_key: str) -> str:
    """Return a cleaned single-line summary from registered attention docstring."""
    cls = ATTENTION_REGISTRY.get(attn_key) if attn_key in ATTENTION_REGISTRY else None
    raw_doc = inspect.getdoc(cls) if cls is not None else ""
    if not raw_doc:
        return f"Specialized attention mechanism `{attn_key}` from the NeuralOps ATTENTION_REGISTRY."
    for marker in ("Attributes:", "Args:", "Parameters:", "Returns:", "Raises:", "Example:", "Examples:", "Note:", "Notes:"):
        idx = raw_doc.find(marker)
        if idx != -1:
            raw_doc = raw_doc[:idx]
    summary = " ".join(raw_doc.split()).strip()
    return summary or f"Specialized attention mechanism `{attn_key}` from the NeuralOps ATTENTION_REGISTRY."


def format_size(num_bytes: int) -> str:
    """Format byte count into human-readable MB/KB string."""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def extract_attention_metadata() -> dict[str, dict[str, str]]:
    """Extract metadata for all registered attention mechanisms."""
    meta: dict[str, dict[str, str]] = {}
    for key, cls in ATTENTION_REGISTRY.items():
        doc = summarize_attention_doc(key)
        family = key.split("_")[0] if "_" in key else "standard"
        module_path = getattr(cls, "__module__", "neuralops.layers")
        meta[key] = {
            "class_name": getattr(cls, "__name__", key),
            "family": family,
            "module_path": module_path,
            "docstring": doc,
        }
    return meta


def collect_artifact_sizes(out_dir: Optional[Path]) -> tuple[list[tuple[str, str]], str]:
    """Scan output directory and compute individual and total artifact sizes."""
    if out_dir is None or not out_dir.exists():
        return [], "0 B"
    rows: list[tuple[str, str]] = []
    total = 0
    for p in sorted(out_dir.glob("*")):
        if p.is_file() and not p.name.startswith("."):
            sz = p.stat().st_size
            total += sz
            rows.append((p.name, format_size(sz)))
    return rows, format_size(total)


def build_qwen_readme(
    attn_key: str,
    result: dict[str, Any],
    out_dir: Optional[Path] = None,
    base_model: str = "Qwen/Qwen2.5-0.5B",
    repo_id: Optional[str] = None,
    num_layers: int = 24,
    dur_s1: Optional[float] = None,
    dur_s2: Optional[float] = None,
    seq_len: int = 512,
    grad_accum: int = 4,
    topk: int = 64,
) -> str:
    """Generate a unified Hugging Face model card README.md for distilled Qwen checkpoints."""
    base_id = normalize_base_model_id(base_model)
    attn_slug = attn_key.replace("_", "-")
    target_repo = repo_id or f"tahamajs/qwen-2.5-0.5b-{attn_slug}-wikitext2"
    pattern = str(result.get("pattern", "hybrid_3_1"))
    if pattern == "hybrid_3_1":
        anchor_layers = [i for i in range(num_layers) if i % 4 == 3]
    else:
        anchor_layers = []
    surgeried_layers = [i for i in range(num_layers) if i not in set(anchor_layers)]

    meta = extract_attention_metadata().get(attn_key, {})
    merged = dict(result)
    if dur_s1 is not None:
        merged["dur_s1"] = round(float(dur_s1), 1)
    if dur_s2 is not None:
        merged["dur_s2"] = round(float(dur_s2), 1)

    return generate_unified_model_card(
        zoo_type="qwen_attention",
        mechanism_key=attn_key,
        repo_id=target_repo,
        base_model=base_id,
        out_dir=out_dir,
        metrics=merged,
        surgery_info={
            "total_layers": num_layers,
            "modified_layers": len(surgeried_layers),
            "pattern": pattern,
            "surgeried_layers": surgeried_layers,
            "anchor_layers": anchor_layers,
            "modified_components": (
                f"`{len(surgeried_layers)}/{num_layers}` decoder `model.layers.{{i}}.self_attn` "
                "+ trained decoder `layernorm`s"
            ),
            "frozen_components": (
                f"`{len(anchor_layers)}` quadratic anchor layers `{anchor_layers}`, "
                "decoder MLPs (`mlp.*`), `embed_tokens`, and `lm_head`"
            ),
            **({k: v for k, v in meta.items() if k in ("class_name", "family", "module_path", "docstring")}),
        },
        dataset="Salesforce/wikitext",
    )


def build_llama_readme(
    attn_key: str,
    result: dict[str, Any],
    teacher_ppl: float,
    out_dir: Path,
    meta: dict[str, Any],
) -> str:
    """Generates a complete TinyLlama-1.1B Hugging Face Model Card via unified generator."""
    slug = attn_key.replace("_", "-")
    repo_id = f"{result.get('hf_org', 'tahamajs')}/tinyllama-1.1b-{slug}-wikitext2"
    pattern = str(result.get("pattern", "hybrid_3_1"))
    total_layers = int(result.get("total_layers", 22))
    surgeried_layers = result.get("surgeried_layers") or (
        [i for i in range(total_layers) if i % 4 != 3] if pattern == "hybrid_3_1" else list(range(total_layers))
    )
    anchor_layers = result.get("anchor_layers") or [i for i in range(total_layers) if i not in set(surgeried_layers)]

    merged_metrics = dict(result)
    if math.isfinite(teacher_ppl):
        merged_metrics["teacher_ppl"] = teacher_ppl

    surgery_info = {
        "pattern": pattern,
        "total_layers": total_layers,
        "modified_layers": len(surgeried_layers),
        "surgeried_layers": surgeried_layers,
        "anchor_layers": anchor_layers,
        "modified_components": (
            f"`{len(surgeried_layers)}/{total_layers}` decoder `model.layers.{{i}}.self_attn` "
            "+ trained decoder `layernorm`s"
        ),
        "frozen_components": (
            f"`{len(anchor_layers)}` quadratic anchor layers `{anchor_layers}`, "
            "decoder MLPs (`mlp.*`), `embed_tokens`, and `lm_head`"
        ),
        **({k: v for k, v in meta.items() if k in ("class_name", "family", "module_path", "docstring")}),
    }

    return generate_unified_model_card(
        zoo_type="tinyllama",
        mechanism_key=attn_key,
        repo_id=repo_id,
        base_model="TinyLlama/TinyLlama_v1.1",
        out_dir=out_dir,
        metrics=merged_metrics,
        surgery_info=surgery_info,
        dataset="Salesforce/wikitext",
    )


def build_vlm_readme(
    attn_key: str,
    result: dict[str, Any],
    teacher_acc: float,
    teacher_ppl: float,
    out_dir: Path,
    meta: dict[str, Any],
) -> str:
    """Generates a complete Hugging Face VLM Model Card via the unified NeuralOps generator."""
    slug = attn_key.replace("_", "-")
    repo_slug = f"llava-qwen-0.5b-{slug}-scienceqa"
    repo_id = f"{result.get('hf_org', 'tahamajs')}/{repo_slug}"
    pattern = str(result.get("pattern", "hybrid_3_1"))
    total_layers = int(result.get("total_layers", 24))
    surgeried_layers = result.get("surgeried_layers") or (
        [i for i in range(total_layers) if i % 4 != 3] if pattern == "hybrid_3_1" else list(range(total_layers))
    )
    anchor_layers = result.get("anchor_layers") or [i for i in range(total_layers) if i not in set(surgeried_layers)]

    merged_metrics = dict(result)
    merged_metrics["teacher_acc"] = teacher_acc
    if math.isfinite(teacher_ppl):
        merged_metrics["teacher_ppl"] = teacher_ppl

    surgery_info = {
        "pattern": pattern,
        "total_layers": total_layers,
        "modified_layers": len(surgeried_layers),
        "surgeried_layers": surgeried_layers,
        "anchor_layers": anchor_layers,
        "modified_components": (
            f"`{len(surgeried_layers)}/{total_layers}` decoder `language_model.model.layers.{{i}}.self_attn` "
            "+ trained decoder `layernorm`s"
        ),
        "frozen_components": (
            f"Frozen SigLIP `vision_tower.*` (27 layers), `multi_modal_projector.*`, "
            f"`{len(anchor_layers)}` quadratic anchor layers `{anchor_layers}`, decoder MLPs & embeddings"
        ),
        **({k: v for k, v in meta.items() if k in ("class_name", "family", "module_path", "docstring")}),
    }

    return generate_unified_model_card(
        zoo_type="llava_vlm",
        mechanism_key=attn_key,
        repo_id=repo_id,
        base_model="llava-hf/llava-interleave-qwen-0.5b-hf",
        out_dir=out_dir,
        metrics=merged_metrics,
        surgery_info=surgery_info,
        dataset="derek-thomas/ScienceQA",
    )


class ZooOrchestrator:
    """
    Multi-module Zoo Execution Orchestrator.

    Handles iterating through attention or compression mechanisms, resume checks,
    checkpoint saving, manifest generation, unified model cards, and ledger logging.
    """

    def __init__(
        self,
        zoo_type: str,
        out_dir: str | Path,
        metrics_log: str | Path,
        hf_org: str = "tahamajs",
        hf_upload: bool = False,
        keep_local_weights: bool = True,
    ) -> None:
        self.zoo_type = zoo_type
        self.out_dir = Path(out_dir)
        self.metrics_log = Path(metrics_log)
        self.hf_org = hf_org
        self.hf_upload = hf_upload
        self.keep_local_weights = keep_local_weights
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_log.parent.mkdir(parents=True, exist_ok=True)

        self.uploader: Optional[AsyncHFHubUploader] = None
        if self.hf_upload:
            self.uploader = AsyncHFHubUploader(
                hf_org=self.hf_org,
                max_workers=2,
                cleanup_weights=not self.keep_local_weights,
            )

    def load_completed_keys(self, key_field: str = "attn_key") -> Set[str]:
        """Read metrics.jsonl to discover mechanisms that already PASSED."""
        completed: Set[str] = set()
        if self.metrics_log.exists():
            try:
                for line in self.metrics_log.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        data = json.loads(line)
                        if data.get("status") == "PASSED":
                            k = data.get(key_field) or data.get("comp_key") or data.get("mechanism_key")
                            if k:
                                completed.add(k)
            except Exception as exc:
                logger.warning("Failed reading metrics log %s: %s", self.metrics_log, exc)
        return completed

    def record_result(self, result: Dict[str, Any]) -> None:
        """Atomically append a benchmark/distillation result line to metrics.jsonl."""
        with open(self.metrics_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")
            fh.flush()

    def run_zoo(
        self,
        keys: List[str],
        train_fn: Callable[[str, Path], Dict[str, Any]],
        key_field: str = "attn_key",
    ) -> List[Dict[str, Any]]:
        """Run training over a sequence of keys with crash-safe resume."""
        completed = self.load_completed_keys(key_field=key_field)
        results: List[Dict[str, Any]] = []

        for i, key in enumerate(keys):
            if key in completed:
                logger.info("[%d/%d] [RESUME] Skipping %s (already PASSED)", i + 1, len(keys), key)
                continue

            ckpt_dir = self.out_dir / key
            try:
                res = train_fn(key, ckpt_dir)
                res["status"] = "PASSED"
                self.record_result(res)
                results.append(res)

                if self.uploader is not None:
                    repo_id = res.get("repo_id") or f"{self.hf_org}/{key}"
                    self.uploader.submit_upload(ckpt_dir, repo_id, mechanism_name=key)
            except Exception as exc:
                logger.error("[%d/%d] Execution failed for %s: %s", i + 1, len(keys), key, exc, exc_info=True)
                fail_res = {
                    key_field: key,
                    "status": "FAILED",
                    "error": str(exc),
                    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                }
                self.record_result(fail_res)
                results.append(fail_res)

        return results


__all__ = [
    "StratifiedQwenDataset",
    "CausalTextDataset",
    "load_tokenized_qwen_corpus",
    "build_cache_wikitext",
    "apply_qwen_attention_surgery",
    "normalize_base_model_id",
    "summarize_attention_doc",
    "format_size",
    "extract_attention_metadata",
    "collect_artifact_sizes",
    "build_qwen_readme",
    "build_llama_readme",
    "build_vlm_readme",
    "ZooOrchestrator",
]
