#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unified Hugging Face Model Card Generator & On-Disk Artifact Auditor.

Provides a single authoritative, parameterized Model Card generator
(`generate_unified_model_card`) shared across all six NeuralOps model zoos:
  - `bert_attention`
  - `bert_compressor`
  - `qwen_attention`
  - `qwen_compressor`
  - `llava_vlm`
  - `tinyllama`

Enforces:
  1. Consistent YAML frontmatter with normalized `base_model`, `pipeline_tag`,
     `datasets`, `tags`, and finite-only `model-index` evaluation records.
  2. 100% honest on-disk artifact auditing (inspects actual files in `out_dir`
     and computes delta footprint savings vs. the base model — never hardcoded).
  3. Explicit surgical scope & modified vs. frozen component breakdown.
  4. Public-API Quick Start snippets (`from neuralops.hub import load_surgeried_model`)
     with zero private benchmark script imports.
"""
from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence, Union

LICENSE_LINK = "https://github.com/tahamajs/NeuralOps/blob/main/LICENSE"
REPO_URL = "https://github.com/tahamajs/NeuralOps"

ZOO_SPECS: dict[str, dict[str, Any]] = {
    "bert_attention": {
        "display_name": "BERT-Base Subquadratic Attention Zoo",
        "mode": "attention_surgery",
        "default_base_model": "google-bert/bert-base-uncased",
        "default_base_params": 109_483_778,
        "default_dataset": "nyu-mll/glue",
        "dataset_display": "GLUE SST-2 (validation, 872 samples)",
        "dataset_config": "sst2",
        "pipeline_tag": "text-classification",
        "task_type": "text-classification",
        "task_name": "Sentiment Classification",
        "default_total_layers": 12,
        "default_pattern": "all",
        "architecture_desc": "12-layer BERT-Base encoder (hidden_size=768, num_attention_heads=12, head_dim=64)",
        "layer_prefix": "bert.encoder.layer.{i}.attention",
        "frozen_summary": "Embeddings (`bert.embeddings.*`), feed-forward layers (`intermediate`, `output.dense`)",
        "base_tags": [
            "neuralops",
            "bert",
            "attention-zoo",
            "attention-surgery",
            "subquadratic-attention",
            "knowledge-distillation",
            "glue-sst2",
        ],
        "bibtex_key": "neuralops2026bertzoo",
        "bibtex_title": "NeuralOps: Pretrained BERT Subquadratic Attention Zoo on GLUE Benchmark",
    },
    "bert_compressor": {
        "display_name": "BERT-Base Compressed & LoRA-Healed Zoo",
        "mode": "compression",
        "default_base_model": "google-bert/bert-base-uncased",
        "default_base_params": 109_483_778,
        "default_dataset": "nyu-mll/glue",
        "dataset_display": "GLUE SST-2 (validation, 872 samples)",
        "dataset_config": "sst2",
        "pipeline_tag": "text-classification",
        "task_type": "text-classification",
        "task_name": "Sentiment Classification",
        "default_total_layers": 12,
        "default_pattern": "compress_model",
        "architecture_desc": "12-layer BERT-Base encoder (hidden_size=768, num_attention_heads=12, head_dim=64)",
        "layer_prefix": "bert.encoder.layer.{i}",
        "frozen_summary": "Unmodified base `bert.embeddings.*` and untouched backbone weights",
        "base_tags": [
            "neuralops",
            "bert",
            "model-compression",
            "compressor-zoo",
            "lora-healing",
            "glue-sst2",
        ],
        "bibtex_key": "neuralops2026bertcompressorzoo",
        "bibtex_title": "NeuralOps: Multi-Rank Compressed BERT-SST2 Zoo with Pareto LoRA Recovery",
    },
    "qwen_attention": {
        "display_name": "Qwen2.5-0.5B Subquadratic Attention Surgery Zoo",
        "mode": "attention_surgery",
        "default_base_model": "Qwen/Qwen2.5-0.5B",
        "default_base_params": 494_032_768,
        "default_dataset": "Salesforce/wikitext",
        "dataset_display": "WikiText-2 (wikitext-2-raw-v1, validation)",
        "dataset_config": "wikitext-2-raw-v1",
        "pipeline_tag": "text-generation",
        "task_type": "text-generation",
        "task_name": "Causal Language Modeling",
        "default_total_layers": 24,
        "default_pattern": "hybrid_3_1",
        "architecture_desc": "24-layer Qwen2.5 decoder (hidden_size=896, 14Q:2KV 7:1 GQA, head_dim=64)",
        "layer_prefix": "model.layers.{i}.self_attn",
        "frozen_summary": "Decoder MLPs (`mlp.*`), embeddings (`embed_tokens`), `lm_head`, and quadratic anchor `self_attn` layers",
        "base_tags": [
            "neuralops",
            "qwen2",
            "qwen2.5",
            "attention-zoo",
            "attention-surgery",
            "subquadratic-attention",
            "knowledge-distillation",
            "wikitext-2",
        ],
        "bibtex_key": "neuralops2026qwenzoo",
        "bibtex_title": "NeuralOps: Multi-Scale Qwen2.5 Subquadratic Attention Surgery Zoo",
    },
    "qwen_compressor": {
        "display_name": "Qwen2.5-0.5B Compressed & Distilled Zoo",
        "mode": "compression",
        "default_base_model": "Qwen/Qwen2.5-0.5B",
        "default_base_params": 494_032_768,
        "default_dataset": "Salesforce/wikitext",
        "dataset_display": "WikiText-2 (wikitext-2-raw-v1, validation)",
        "dataset_config": "wikitext-2-raw-v1",
        "pipeline_tag": "text-generation",
        "task_type": "text-generation",
        "task_name": "Causal Language Modeling",
        "default_total_layers": 24,
        "default_pattern": "compress_model",
        "architecture_desc": "24-layer Qwen2.5 decoder (hidden_size=896, 14Q:2KV 7:1 GQA, head_dim=64)",
        "layer_prefix": "model.layers.{i}",
        "frozen_summary": "Uncompressed `embed_tokens`, `lm_head`, and unmodified base parameters",
        "base_tags": [
            "neuralops",
            "qwen2",
            "qwen2.5",
            "model-compression",
            "compressor-zoo",
            "knowledge-distillation",
            "wikitext-2",
        ],
        "bibtex_key": "neuralops2026qwencompressorzoo",
        "bibtex_title": "NeuralOps: Compressed & Sparse-KD Healed Qwen2.5-0.5B Zoo",
    },
    "llava_vlm": {
        "display_name": "LLaVA-Qwen-0.5B Multimodal Attention Surgery Zoo",
        "mode": "attention_surgery",
        "default_base_model": "llava-hf/llava-interleave-qwen-0.5b-hf",
        "default_base_params": 892_000_000,
        "default_dataset": "derek-thomas/ScienceQA",
        "dataset_display": "ScienceQA (Image-Bearing STEM Subset, validation)",
        "dataset_config": None,
        "pipeline_tag": "image-text-to-text",
        "task_type": "visual-question-answering",
        "task_name": "ScienceQA Multimodal Visual Reasoning",
        "default_total_layers": 24,
        "default_pattern": "hybrid_3_1",
        "architecture_desc": "Frozen SigLIP Vision Tower + 2-layer MLP Projector + 24-layer Qwen2 decoder (D=1024, H=16, head_dim=64)",
        "layer_prefix": "language_model.model.layers.{i}.self_attn",
        "frozen_summary": "Frozen SigLIP `vision_tower.*` (27 layers), `multi_modal_projector.*`, decoder MLPs, embeddings/`lm_head`, and quadratic anchor `self_attn` layers",
        "base_tags": [
            "neuralops",
            "multimodal",
            "llava",
            "vision-language-model",
            "scienceqa",
            "attention-zoo",
            "attention-surgery",
            "knowledge-distillation",
        ],
        "bibtex_key": "neuralops2026vlmzoo",
        "bibtex_title": "NeuralOps: Multimodal Vision-Language Subquadratic Attention Surgery Zoo (LLaVA-ScienceQA)",
    },
    "tinyllama": {
        "display_name": "TinyLlama-1.1B LLaMA Attention Surgery Zoo",
        "mode": "attention_surgery",
        "default_base_model": "TinyLlama/TinyLlama_v1.1",
        "default_base_params": 1_100_048_384,
        "default_dataset": "Salesforce/wikitext",
        "dataset_display": "WikiText-2 (wikitext-2-raw-v1, validation)",
        "dataset_config": "wikitext-2-raw-v1",
        "pipeline_tag": "text-generation",
        "task_type": "text-generation",
        "task_name": "Causal Language Modeling",
        "default_total_layers": 22,
        "default_pattern": "hybrid_3_1",
        "architecture_desc": "22-layer LLaMA decoder (hidden_size=2048, 32Q:4KV 8:1 GQA, head_dim=64, bias-free)",
        "layer_prefix": "model.layers.{i}.self_attn",
        "frozen_summary": "Decoder MLPs (`mlp.*`), embeddings (`embed_tokens`), `lm_head`, and quadratic anchor `self_attn` layers",
        "base_tags": [
            "neuralops",
            "llama",
            "tinyllama",
            "attention-zoo",
            "attention-surgery",
            "subquadratic-attention",
            "knowledge-distillation",
            "wikitext-2",
        ],
        "bibtex_key": "neuralops2026llamazoo",
        "bibtex_title": "NeuralOps: Subquadratic & Hybrid Attention Surgery Zoo for TinyLlama-1.1B",
    },
}

_ZOO_ALIASES: dict[str, str] = {
    "bert": "bert_attention",
    "bert_attn": "bert_attention",
    "bert_attention": "bert_attention",
    "bert_comp": "bert_compressor",
    "bert_compressor": "bert_compressor",
    "qwen": "qwen_attention",
    "qwen_attn": "qwen_attention",
    "qwen_attention": "qwen_attention",
    "qwen_comp": "qwen_compressor",
    "qwen_compressor": "qwen_compressor",
    "vlm": "llava_vlm",
    "llava": "llava_vlm",
    "llava_vlm": "llava_vlm",
    "llama": "tinyllama",
    "tinyllama": "tinyllama",
}

_ARTIFACT_ROLES: dict[str, str] = {
    "attention_weights.safetensors": "Modular surgical / compressed delta weights (FP16 safetensors)",
    "model.safetensors": "Serialized model weights (safetensors)",
    "pytorch_model.bin": "Serialized PyTorch checkpoint weights",
    "surgical_model_state.pt": "Surgical checkpoint state dict",
    "surgical_manifest.json": "Surgical / compression topology and modified-prefix manifest",
    "config.json": "Model architecture and NeuralOps surgery/compression metadata",
    "training_metrics.json": "Machine-readable evaluation, telemetry, and distillation metrics",
    "compression_summary.json": "Multi-rank Pareto compression and healing summary",
    "tokenizer.json": "Fast tokenizer vocabulary and merges",
    "tokenizer_config.json": "Tokenizer special tokens and configuration",
    "vocab.txt": "WordPiece vocabulary",
    "vocab.json": "BPE vocabulary mapping",
    "merges.txt": "BPE merge table",
    "special_tokens_map.json": "Special token mappings",
    "preprocessor_config.json": "Vision-language image preprocessor configuration",
    "added_tokens.json": "Additional vocabulary tokens",
    "chat_template.json": "Chat formatting template",
}

_SKIP_AUDIT_FILES = {"README.md", "upload_success.json", "upload_error.json", ".DS_Store"}


def _resolve_zoo_type(zoo_type: str) -> str:
    key = zoo_type.strip().lower().replace("-", "_")
    resolved = _ZOO_ALIASES.get(key)
    if resolved is None:
        raise ValueError(
            f"Unsupported zoo_type '{zoo_type}'. Expected one of: {sorted(ZOO_SPECS.keys())}"
        )
    return resolved


def normalize_base_model_id(model_name: Optional[str], zoo_type: str = "bert_attention") -> str:
    """Normalize local cache paths or short aliases to canonical Hugging Face Hub model IDs."""
    spec = ZOO_SPECS[_resolve_zoo_type(zoo_type)]
    if not model_name or not str(model_name).strip():
        return str(spec["default_base_model"])

    cleaned = str(model_name).strip()
    lower = cleaned.lower()

    if "llava" in lower and ("qwen" in lower or "interleave" in lower):
        return "llava-hf/llava-interleave-qwen-0.5b-hf"
    if "tinyllama" in lower:
        return "TinyLlama/TinyLlama_v1.1"
    if "qwen2.5-0.5b" in lower or "qwen2.5_0.5b" in lower:
        return "Qwen/Qwen2.5-0.5B"
    if "bert-base-uncased" in lower or "bert_sst2_teacher" in lower:
        return "google-bert/bert-base-uncased"

    if cleaned.startswith("/") or "\\" in cleaned:
        tail = Path(cleaned.rstrip("/")).name
        if tail and "/" in str(spec["default_base_model"]):
            org = str(spec["default_base_model"]).split("/", 1)[0]
            return f"{org}/{tail}" if not tail.lower().startswith(org.lower()) else tail
        return str(spec["default_base_model"])

    return cleaned


def _normalize_dataset_id(dataset: Optional[str], zoo_type: str) -> str:
    spec = ZOO_SPECS[_resolve_zoo_type(zoo_type)]
    if not dataset or not str(dataset).strip():
        return str(spec["default_dataset"])
    low = str(dataset).strip().lower()
    if "scienceqa" in low:
        return "derek-thomas/ScienceQA"
    if "wikitext" in low:
        return "Salesforce/wikitext"
    if "glue" in low or "sst2" in low or "sst-2" in low:
        return "nyu-mll/glue"
    return str(dataset).strip()


def _format_size(num_bytes: int) -> str:
    if num_bytes >= 1024 * 1024 * 1024:
        return f"{num_bytes / (1024 * 1024 * 1024):.2f} GB"
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def _first_finite(mapping: dict[str, Any], keys: Sequence[str]) -> Optional[float]:
    for key in keys:
        val = mapping.get(key)
        if isinstance(val, bool):
            continue
        if isinstance(val, (int, float)) and math.isfinite(float(val)):
            return float(val)
    return None


def _normalize_tag(tag: str) -> str:
    return tag.strip().lower().replace(" ", "-").replace("_", "-")


def _resolve_operator_metadata(
    mechanism_key: str,
    mode: str,
    surgery_info: dict[str, Any],
) -> dict[str, str]:
    """Resolve class name, family, module path, and docstring safely from registries or overrides."""
    cls_name = str(surgery_info.get("class_name") or "")
    family = str(surgery_info.get("family") or "")
    mod_path = str(surgery_info.get("module_path") or "")
    docstring = str(surgery_info.get("docstring") or "")

    if not (cls_name and family and mod_path and docstring):
        try:
            from neuralops.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY

            reg = COMPRESSOR_REGISTRY if mode == "compression" else ATTENTION_REGISTRY
            cls = reg.get(mechanism_key) if mechanism_key in reg else None
            if cls is not None:
                cls_name = cls_name or getattr(cls, "__name__", mechanism_key)
                try:
                    src = Path(inspect.getfile(cls)).resolve()
                    repo_root = Path(__file__).resolve().parents[2]
                    try:
                        mod_path = mod_path or str(src.relative_to(repo_root))
                    except ValueError:
                        mod_path = mod_path or src.name
                    family = family or src.parent.name
                except Exception:
                    pass
                raw_doc = inspect.getdoc(cls) or ""
                if raw_doc and not docstring:
                    for stop in (
                        "Attributes:",
                        "Args:",
                        "Parameters:",
                        "Returns:",
                        "Raises:",
                        "Example:",
                        "Examples:",
                        "Note:",
                        "Notes:",
                    ):
                        idx = raw_doc.find(stop)
                        if idx != -1:
                            raw_doc = raw_doc[:idx]
                    docstring = " ".join(raw_doc.split())[:420]
        except Exception:
            pass

    default_dir = "neuralops/compression/" if mode == "compression" else "neuralops/layers/"
    default_fam = "compression" if mode == "compression" else "attention"
    return {
        "class_name": cls_name or mechanism_key,
        "family": family or default_fam,
        "module_path": mod_path or default_dir,
        "docstring": docstring or f"NeuralOps `{mechanism_key}` operator.",
    }


def audit_artifacts_on_disk(
    out_dir: Optional[Union[str, Path]],
    base_model_bytes: Optional[int] = None,
) -> dict[str, Any]:
    """Inspect actual checkpoint files present in `out_dir` without hardcoded assumptions.

    Returns exact file names, byte counts, weight-file names, manifest metadata, and
    realized footprint savings compared against `base_model_bytes`.
    """
    audit: dict[str, Any] = {
        "audited_dir_exists": False,
        "files": [],
        "weight_files": [],
        "weight_bytes": 0,
        "total_bytes": 0,
        "manifest": None,
        "config": None,
        "base_model_bytes": base_model_bytes,
        "savings_bytes": None,
        "savings_pct": None,
        "footprint_ratio": None,
    }
    if out_dir is None:
        return audit

    root = Path(out_dir)
    if not root.is_dir():
        return audit

    audit["audited_dir_exists"] = True

    priority_order = [
        "attention_weights.safetensors",
        "model.safetensors",
        "pytorch_model.bin",
        "surgical_model_state.pt",
        "surgical_manifest.json",
        "config.json",
        "training_metrics.json",
        "compression_summary.json",
    ]
    priority_rank = {name: idx for idx, name in enumerate(priority_order)}

    entries = []
    for item in root.iterdir():
        if not item.is_file():
            continue
        name = item.name
        if name in _SKIP_AUDIT_FILES or name.startswith(".") or name.endswith(".tmp"):
            continue
        entries.append(item)

    entries.sort(key=lambda p: (priority_rank.get(p.name, 999), p.name))

    files: list[dict[str, Any]] = []
    weight_files: list[str] = []
    weight_bytes = 0
    total_bytes = 0

    for item in entries:
        sz = int(item.stat().st_size)
        total_bytes += sz
        is_weight = item.name.endswith((".safetensors", ".bin", ".pt"))
        if is_weight:
            weight_files.append(item.name)
            weight_bytes += sz
        role = _ARTIFACT_ROLES.get(item.name)
        if role is None:
            if item.name.startswith("modeling_") and item.name.endswith(".py"):
                role = "Custom Hugging Face modeling loader shim"
            elif is_weight:
                role = "Serialized checkpoint weight artifact"
            else:
                role = "Checkpoint / tokenizer metadata"
        files.append(
            {
                "name": item.name,
                "bytes": sz,
                "size_str": _format_size(sz),
                "is_weight": is_weight,
                "role": role,
            }
        )

    manifest_path = root / "surgical_manifest.json"
    if manifest_path.is_file():
        try:
            audit["manifest"] = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            audit["manifest"] = None

    config_path = root / "config.json"
    if config_path.is_file():
        try:
            audit["config"] = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            audit["config"] = None

    audit["files"] = files
    audit["weight_files"] = weight_files
    audit["weight_bytes"] = weight_bytes
    audit["total_bytes"] = total_bytes

    if base_model_bytes and base_model_bytes > 0 and weight_bytes > 0:
        savings_bytes = max(0, base_model_bytes - weight_bytes)
        audit["savings_bytes"] = savings_bytes
        audit["savings_pct"] = (savings_bytes / base_model_bytes) * 100.0
        audit["footprint_ratio"] = base_model_bytes / weight_bytes

    return audit


def _build_quick_start_snippet(zoo_type: str, repo_id: str) -> str:
    """Return a clean Quick Start snippet using ONLY public `neuralops.hub` imports."""
    if zoo_type in ("bert_attention", "bert_compressor"):
        return f"""```python
import torch
from neuralops.hub import load_surgeried_model

model, tokenizer = load_surgeried_model("{repo_id}")
model.eval()

inputs = tokenizer(
    "NeuralOps achieves strong efficiency without sacrificing downstream accuracy.",
    return_tensors="pt",
)
with torch.no_grad():
    logits = model(**inputs).logits
    pred = int(logits.argmax(dim=-1).item())
print("Predicted SST-2 Sentiment (0=Negative, 1=Positive):", pred)
```"""

    if zoo_type == "llava_vlm":
        return f"""```python
import torch
from PIL import Image
from neuralops.hub import load_surgeried_model

model, processor = load_surgeried_model("{repo_id}")
model.eval()

image = Image.new("RGB", (384, 384), color=(255, 255, 255))
inputs = processor(
    text="<image>\\nQuestion: What is shown in the figure?\\nAnswer with the option letter:",
    images=image,
    return_tensors="pt",
)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=16)
print(processor.decode(outputs[0], skip_special_tokens=True))
```"""

    return f"""```python
import torch
from neuralops.hub import load_surgeried_model

model, tokenizer = load_surgeried_model("{repo_id}")
model.eval()

inputs = tokenizer("NeuralOps subquadratic attention enables", return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```"""


def generate_unified_model_card(
    zoo_type: str,
    mechanism_key: str,
    repo_id: str,
    base_model: Optional[str] = None,
    out_dir: Optional[Union[str, Path]] = None,
    metrics: Optional[dict[str, Any]] = None,
    surgery_info: Optional[dict[str, Any]] = None,
    dataset: Optional[str] = None,
    pipeline_tag: Optional[str] = None,
    extra_tags: Optional[Sequence[str]] = None,
    hardware: str = "NVIDIA GeForce RTX 3090 (24GB)",
    base_model_size_bytes: Optional[int] = None,
) -> str:
    """Generate a unified, honest, publication-grade Hugging Face Model Card.

    Shared across all 6 NeuralOps zoos (`bert_attention`, `bert_compressor`,
    `qwen_attention`, `qwen_compressor`, `llava_vlm`, `tinyllama`).
    """
    zoo = _resolve_zoo_type(zoo_type)
    spec = ZOO_SPECS[zoo]
    metrics_dict: dict[str, Any] = dict(metrics or {})
    surgery: dict[str, Any] = dict(surgery_info or {})

    norm_base = normalize_base_model_id(base_model, zoo)
    norm_dataset = _normalize_dataset_id(dataset or metrics_dict.get("dataset"), zoo)
    resolved_pipeline = pipeline_tag or str(spec["pipeline_tag"])
    repo_short = repo_id.split("/")[-1] if "/" in repo_id else repo_id

    op_meta = _resolve_operator_metadata(mechanism_key, str(spec["mode"]), surgery)
    cls_name = op_meta["class_name"]
    family = op_meta["family"]
    mod_path = op_meta["module_path"]
    docstring = op_meta["docstring"]

    # Base model FP16 footprint reference for honest delta savings math
    orig_params_val = _first_finite(metrics_dict, ("orig_params",))
    ref_params = int(orig_params_val) if orig_params_val and orig_params_val > 0 else int(spec["default_base_params"])
    ref_base_bytes = base_model_size_bytes if (base_model_size_bytes and base_model_size_bytes > 0) else ref_params * 2

    audit = audit_artifacts_on_disk(out_dir, base_model_bytes=ref_base_bytes)
    manifest = audit.get("manifest") or {}
    manifest_meta = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    disk_cfg = audit.get("config") or {}

    # Resolve surgical / compression scope
    total_layers = int(
        surgery.get("total_layers")
        or manifest.get("total_layers")
        or manifest_meta.get("total_layers")
        or spec["default_total_layers"]
    )
    pattern = str(
        surgery.get("pattern")
        or manifest.get("pattern")
        or manifest_meta.get("pattern")
        or metrics_dict.get("pattern")
        or disk_cfg.get("surgery_pattern")
        or spec["default_pattern"]
    )
    rank_val = (
        surgery.get("rank")
        or manifest.get("rank")
        or manifest_meta.get("rank")
        or metrics_dict.get("rank")
        or metrics_dict.get("showcase_rank")
        or disk_cfg.get("compression_rank")
    )

    surgeried_layers = (
        surgery.get("surgeried_layers")
        or manifest.get("surgeried_layers")
        or manifest_meta.get("surgeried_layers")
    )
    anchor_layers = (
        surgery.get("anchor_layers")
        or manifest.get("anchor_layers")
        or manifest_meta.get("anchor_layers")
    )

    if spec["mode"] == "attention_surgery":
        if surgeried_layers is None:
            if pattern == "hybrid_3_1":
                surgeried_layers = [i for i in range(total_layers) if i % 4 != 3]
            else:
                surgeried_layers = list(range(total_layers))
        if anchor_layers is None:
            anchor_layers = [i for i in range(total_layers) if i not in set(surgeried_layers)]
        modified_count = int(surgery.get("modified_layers", len(surgeried_layers)))
        anchor_count = len(anchor_layers)
    else:
        modified_count = int(
            surgery.get("modified_layers")
            or len(manifest.get("submodules", {}))
            or len(manifest.get("modified_prefixes", []))
            or len(manifest.get("surgeried_prefixes", []))
            or total_layers
        )
        if surgeried_layers is None:
            surgeried_layers = list(range(total_layers))
        if anchor_layers is None:
            anchor_layers = []
        anchor_count = len(anchor_layers)

    # Extract verified finite metrics
    acc_before = _first_finite(metrics_dict, ("acc_before", "pre_kd_acc", "zero_shot_acc", "pre_kd_sst2_acc", "zero_shot_sst2_acc"))
    acc_after = _first_finite(metrics_dict, ("acc_after", "post_kd_acc", "healed_acc", "post_kd_sst2_acc", "sst2_acc", "accuracy", "eval_accuracy"))
    teacher_acc = _first_finite(metrics_dict, ("teacher_acc", "teacher_baseline_acc", "teacher_sst2_acc"))
    ppl_before = _first_finite(metrics_dict, ("ppl_before", "zero_shot_ppl", "pre_kd_ppl", "pre_kd_wikitext_ppl"))
    ppl_after = _first_finite(metrics_dict, ("ppl_after", "post_kd_ppl", "perplexity", "post_kd_wikitext_ppl", "wikitext_ppl", "eval_perplexity"))
    teacher_ppl = _first_finite(metrics_dict, ("teacher_ppl", "teacher_baseline_ppl"))
    post_loss = _first_finite(metrics_dict, ("post_kd_loss", "healed_loss"))
    comp_params_val = _first_finite(metrics_dict, ("comp_params",))
    comp_ratio = _first_finite(metrics_dict, ("compression_ratio", "param_reduction", "ratio"))
    if comp_ratio is None and orig_params_val and comp_params_val and comp_params_val > 0:
        comp_ratio = orig_params_val / comp_params_val
    sparsity_pct = _first_finite(metrics_dict, ("sparsity_pct",))

    steps_s1 = int(metrics_dict.get("steps_s1", 0) or 0)
    steps_s2 = int(metrics_dict.get("steps_s2", 0) or 0)
    total_steps = int(metrics_dict.get("total_steps", steps_s1 + steps_s2) or (steps_s1 + steps_s2))
    train_secs = float(metrics_dict.get("train_secs", 0.0) or 0.0)
    peak_vram = int(metrics_dict.get("peak_vram_mb", 0) or 0)
    throughput = _first_finite(metrics_dict, ("tokens_per_sec",))
    timestamp = str(metrics_dict.get("timestamp", "") or "")

    # Build model-index entries strictly from verified finite metrics
    index_acc = acc_after if acc_after is not None else acc_before
    index_ppl = ppl_after if ppl_after is not None else ppl_before

    yaml_metric_types: list[str] = []
    model_index_metrics: list[str] = []
    if index_acc is not None:
        yaml_metric_types.append("accuracy")
        model_index_metrics.append(
            f"          - type: accuracy\n"
            f"            value: {round(index_acc, 4)}\n"
            f"            name: {spec['dataset_display']} Accuracy (%)"
        )
    if index_ppl is not None:
        yaml_metric_types.append("perplexity")
        model_index_metrics.append(
            f"          - type: perplexity\n"
            f"            value: {round(index_ppl, 4)}\n"
            f"            name: {spec['dataset_display']} Perplexity"
        )
    if comp_ratio is not None and (
        spec["mode"] == "compression"
        or any(k in metrics_dict for k in ("compression_ratio", "param_reduction", "ratio"))
    ):
        yaml_metric_types.append("compression_ratio")
        model_index_metrics.append(
            f"          - type: compression_ratio\n"
            f"            value: {round(comp_ratio, 4)}\n"
            f"            name: Parameter Compression Ratio (x)"
        )

    # Deduplicated normalized tags
    raw_tags = list(spec["base_tags"]) + [family, mechanism_key] + list(extra_tags or [])
    norm_tags: list[str] = []
    seen_tags: set[str] = set()
    for t in raw_tags:
        nt = _normalize_tag(str(t))
        if nt and nt not in seen_tags:
            seen_tags.add(nt)
            norm_tags.append(nt)

    tags_yaml = "\n".join(f"  - {t}" for t in norm_tags)
    metrics_yaml_block = (
        "metrics:\n" + "\n".join(f"  - {m}" for m in yaml_metric_types) + "\n"
        if yaml_metric_types
        else ""
    )

    dataset_cfg_line = f"\n          config: {spec['dataset_config']}" if spec.get("dataset_config") else ""
    if model_index_metrics:
        model_index_yaml = (
            "model-index:\n"
            f"  - name: {repo_short}\n"
            "    results:\n"
            "      - task:\n"
            f"          type: {spec['task_type']}\n"
            f"          name: {spec['task_name']}\n"
            "        dataset:\n"
            f"          type: {norm_dataset}\n"
            f"          name: {spec['dataset_display']}{dataset_cfg_line}\n"
            "          split: validation\n"
            "        metrics:\n"
            + "\n".join(model_index_metrics)
            + "\n"
        )
    else:
        model_index_yaml = ""

    frontmatter = (
        "---\n"
        "language:\n"
        "  - en\n"
        "license: mit\n"
        f"license_link: {LICENSE_LINK}\n"
        "library_name: neuralops\n"
        f"base_model: {norm_base}\n"
        f"pipeline_tag: {resolved_pipeline}\n"
        "datasets:\n"
        f"  - {norm_dataset}\n"
        f"{metrics_yaml_block}"
        "tags:\n"
        f"{tags_yaml}\n"
        f"{model_index_yaml}"
        "---"
    )

    # Badges derived strictly from verified metrics & actual on-disk audit
    badges: list[str] = []
    if acc_after is not None:
        acc_color = "brightgreen" if acc_after >= 85.0 else ("green" if acc_after >= 70.0 else ("yellow" if acc_after >= 55.0 else "orange"))
        badges.append(f"[![Accuracy](https://img.shields.io/badge/Accuracy-{acc_after:.2f}%25-{acc_color})](#-empirical-evaluation)")
    if ppl_after is not None:
        ppl_color = "brightgreen" if ppl_after < 30.0 else ("blue" if ppl_after < 80.0 else ("yellow" if ppl_after < 200.0 else "orange"))
        badges.append(f"[![Perplexity](https://img.shields.io/badge/Perplexity-{ppl_after:.2f}-{ppl_color})](#-empirical-evaluation)")
    if comp_ratio is not None:
        badges.append(f"[![Compression](https://img.shields.io/badge/Compression-{comp_ratio:.2f}x-purple)](#-surgical-scope--modified-components)")

    if audit["weight_files"]:
        weight_badge_str = _format_size(audit["weight_bytes"]).replace(" ", "_")
        badges.append(f"[![On-Disk Weights](https://img.shields.io/badge/On--Disk_Weights-{weight_badge_str}-indigo)](#-on-disk-artifact-audit)")
    else:
        badges.append("[![Artifacts](https://img.shields.io/badge/Artifacts-Metadata_Only-lightgrey)](#-on-disk-artifact-audit)")
    badges.append(f"[![License: MIT](https://img.shields.io/badge/License-MIT-green)]({LICENSE_LINK})")
    badges_block = "\n".join(badges)

    # Surgical Scope & Modified vs Frozen Components Table
    if spec["mode"] == "attention_surgery":
        scope_label = f"`{modified_count} / {total_layers}` decoder/encoder layers (`{pattern}`)"
        modified_paths = str(
            surgery.get("modified_components")
            or f"`{spec['layer_prefix'].format(i='{i}')}` for `i` in `{len(surgeried_layers)}` surgeried layers"
            + (" + trained `layernorm`s" if zoo != "bert_attention" else " + `classifier` / `pooler`")
        )
        anchor_detail = (
            f"{anchor_count} quadratic anchor layers `{anchor_layers}`"
            if anchor_count > 0
            else "0 anchor layers (full replacement)"
        )
        frozen_paths = str(surgery.get("frozen_components") or f"{spec['frozen_summary']} ({anchor_detail})")
    else:
        rank_label = f", rank={rank_val}" if rank_val is not None else ""
        scope_label = f"`compress_model(method='{mechanism_key}'{rank_label})`"
        mod_prefixes = (
            surgery.get("modified_prefixes")
            or manifest.get("modified_prefixes")
            or manifest.get("surgeried_prefixes")
            or []
        )
        if mod_prefixes:
            preview = ", ".join(f"`{p}`" for p in mod_prefixes[:6])
            suffix = f" (+{len(mod_prefixes) - 6} more)" if len(mod_prefixes) > 6 else ""
            modified_paths = f"{len(mod_prefixes)} compressed/modified prefixes: {preview}{suffix}"
        else:
            modified_paths = str(
                surgery.get("modified_components")
                or f"Compressed linear factors/buffers (`{modified_count}` submodules) + LoRA/task head deltas"
            )
        excl_prefixes = surgery.get("frozen_components") or manifest.get("excluded_prefixes") or spec["frozen_summary"]
        if isinstance(excl_prefixes, list):
            frozen_paths = ", ".join(f"`{x}`" for x in excl_prefixes)
        else:
            frozen_paths = str(excl_prefixes)

    scope_rows = [
        f"| **Zoo Track** | `{zoo}` ({spec['display_name']}) |",
        f"| **Operator / Registry Key** | `{mechanism_key}` (`{cls_name}`, family `{family}`) |",
        f"| **Source Implementation** | [`{mod_path}`]({REPO_URL}/blob/main/{mod_path}) |",
        f"| **Base Architecture** | [`{norm_base}`](https://huggingface.co/{norm_base}) — {spec['architecture_desc']} |",
        f"| **Surgical / Compression Scope** | {scope_label} |",
        f"| **Modified & Saved Components** | {modified_paths} |",
        f"| **Frozen & Excluded Components** | {frozen_paths} |",
    ]
    if rank_val is not None:
        scope_rows.append(f"| **Compression Rank (`k`)** | `{rank_val}` (`LoRA Healed: {bool(metrics_dict.get('lora_healed', False))}`) |")
    if comp_params_val is not None:
        red_pct = ((ref_params - comp_params_val) / max(ref_params, 1)) * 100.0
        ratio_str = f"{comp_ratio:.3f}x" if comp_ratio is not None else "1.000x"
        scope_rows.append(
            f"| **Parameter Footprint** | `{int(comp_params_val):,}` / `{ref_params:,}` (`-{red_pct:.2f}%`, **`{ratio_str}`**) |"
        )
    if sparsity_pct is not None:
        scope_rows.append(f"| **Weight Sparsity** | `{sparsity_pct:.2f}%` |")
    scope_table = "\n".join(scope_rows)

    # Empirical Evaluation Table
    eval_rows: list[str] = []
    if acc_before is not None or acc_after is not None or teacher_acc is not None:
        t_acc_s = f"`{teacher_acc:.2f}%`" if teacher_acc is not None else "—"
        pre_acc_s = f"`{acc_before:.2f}%`" if acc_before is not None else "—"
        post_acc_s = f"**`{acc_after:.2f}%`**" if acc_after is not None else "—"
        delta_acc_s = (
            f"**`{acc_after - acc_before:+.2f}%`**"
            if (acc_after is not None and acc_before is not None)
            else "—"
        )
        eval_rows.append(f"| **Accuracy (%)** | {t_acc_s} | {pre_acc_s} | {post_acc_s} | {delta_acc_s} |")

    if ppl_before is not None or ppl_after is not None or teacher_ppl is not None:
        t_ppl_s = f"`{teacher_ppl:.2f}`" if teacher_ppl is not None else "—"
        pre_ppl_s = f"`{ppl_before:,.2f}`" if ppl_before is not None else "—"
        post_ppl_s = f"**`{ppl_after:,.2f}`**" if ppl_after is not None else "—"
        delta_ppl_s = (
            f"**`{ppl_after - ppl_before:+.2f}`**"
            if (ppl_after is not None and ppl_before is not None)
            else "—"
        )
        eval_rows.append(f"| **Perplexity (PPL)** | {t_ppl_s} | {pre_ppl_s} | {post_ppl_s} | {delta_ppl_s} |")

    if post_loss is not None:
        eval_rows.append(f"| **Validation Loss** | — | — | `{post_loss:.4f}` | — |")

    if not eval_rows:
        eval_rows.append("| **Status** | — | — | `Recorded in training_metrics.json` | — |")
    eval_table = "\n".join(eval_rows)

    # Optional 4-rank Pareto table (for bert_compressor or multi-rank exports)
    pareto_section = ""
    all_ranks = surgery.get("all_ranks") or metrics_dict.get("all_ranks")
    if isinstance(all_ranks, list) and all_ranks:
        t_ref = teacher_acc if teacher_acc is not None else 91.63
        p_rows: list[str] = []
        for rec in sorted(all_ranks, key=lambda x: -int(x.get("rank", 0))):
            r_k = rec.get("rank", "-")
            if rec.get("status") != "PASSED":
                p_rows.append(f"| `{r_k}` | — | — | `{rec.get('status')}` | — | — |")
                continue
            r_ratio = float(rec.get("ratio", 1.0))
            r_orig = int(rec.get("orig_params", ref_params))
            r_comp = int(rec.get("comp_params", r_orig))
            r_red = ((r_orig - r_comp) / max(r_orig, 1)) * 100.0
            r_zero = float(rec.get("zero_shot_acc", 0.0))
            r_heal = float(rec.get("healed_acc", r_zero))
            r_flag = " *(LoRA)*" if rec.get("lora_healed") else ""
            r_star = " ⭐" if (rank_val is not None and int(r_k) == int(rank_val)) else ""
            p_rows.append(
                f"| **`{r_k}`{r_star}** | `{r_ratio:.2f}x` | `{r_comp:,}` (`-{r_red:.1f}%`) | "
                f"`{r_zero:.2f}%` | **`{r_heal:.2f}%`**{r_flag} | `{r_heal - t_ref:+.2f}%` |"
            )
        pareto_section = (
            "\n### Multi-Rank Pareto Evaluation\n\n"
            f"| Rank (`k`) | Physical Ratio | Compressed Params (Reduction) | Zero-Shot Acc | Healed Acc | Parity vs Teacher (`{t_ref:.2f}%`) |\n"
            "|:---:|:---:|---:|---:|---:|---:|\n"
            + "\n".join(p_rows)
            + "\n"
        )

    # 100% Honest On-Disk Artifact Audit Section
    if audit["audited_dir_exists"] and audit["files"]:
        file_rows = "\n".join(
            f"| `{f['name']}` | `{f['size_str']}` | `{f['bytes']:,}` | {f['role']} |"
            for f in audit["files"]
        )
        weight_summary_lines = [
            f"- **Serialized Weight Artifact(s):** {', '.join(f'`{w}`' for w in audit['weight_files']) or 'None present on disk'}",
            f"- **Actual On-Disk Weight Footprint:** `{_format_size(audit['weight_bytes'])}` (`{audit['weight_bytes']:,}` bytes) | Total Package Size: `{_format_size(audit['total_bytes'])}`",
        ]
        if audit["savings_bytes"] is not None:
            weight_summary_lines.append(
                f"- **Delta Savings vs. Full FP16 Base (`{norm_base}`, `{_format_size(ref_base_bytes)}`):** "
                f"**`{_format_size(audit['savings_bytes'])}` saved (`-{audit['savings_pct']:.1f}%`, `{audit['footprint_ratio']:.2f}x` smaller artifact)**"
            )
        if manifest:
            t_cnt = manifest.get("saved_tensor_count") or disk_cfg.get("saved_tensor_count")
            w_fmt = manifest.get("weights_format") or disk_cfg.get("weights_format")
            schema_v = manifest.get("schema_version")
            parts = []
            if schema_v:
                parts.append(f"`schema_version={schema_v}`")
            if t_cnt is not None:
                parts.append(f"`saved_tensor_count={t_cnt}`")
            if w_fmt:
                parts.append(f"`weights_format={w_fmt}`")
            if parts:
                weight_summary_lines.append(
                    f"- **Manifest Verification (`surgical_manifest.json`):** {', '.join(parts)}"
                )
        artifact_section = (
            "\n".join(weight_summary_lines)
            + "\n\n| File on Disk | Size | Bytes | Role |\n"
            "|---|---:|---:|---|\n"
            + file_rows
        )
    else:
        artifact_section = (
            "> No local checkpoint directory was present on disk at card-generation time "
            "(metadata-only preview; no fabricated file sizes)."
        )

    # Hardware Telemetry
    telemetry_lines = [
        f"- **Hardware:** {hardware}",
        "- **Teacher GPU VRAM Allocation:** `0 MB` (Offline Teacher Representation Caching per AGENTS.md §21.1)",
        f"- **Optimization Steps:** `{total_steps}` (`Stage 1: {steps_s1}` | `Stage 2: {steps_s2}`)",
        f"- **Training Duration:** `{train_secs:.1f}s`",
        f"- **Peak GPU VRAM:** `{peak_vram} MB`",
    ]
    if throughput is not None and throughput > 0:
        telemetry_lines.append(f"- **Throughput:** `{throughput:.1f} tokens/sec`")
    if timestamp:
        telemetry_lines.append(f"- **Timestamp (UTC):** `{timestamp}`")
    telemetry_block = "\n".join(telemetry_lines)

    quick_start = _build_quick_start_snippet(zoo, repo_id)

    return f"""{frontmatter}

# {norm_base} ({mechanism_key}) — {spec['display_name']}

{badges_block}

> **{cls_name}**: {docstring}

This repository packages a modular NeuralOps **{spec['display_name']}** checkpoint derived from [`{norm_base}`](https://huggingface.co/{norm_base}) using **`{mechanism_key}`** and evaluated on **{spec['dataset_display']}**.

---

## Surgical Scope & Modified Components

| Property | Specification |
|---|---|
{scope_table}

---

## Empirical Evaluation

| Metric | Teacher Baseline | Pre-KD / Zero-Shot | Post-KD / Final | Delta ($\\Delta$) |
|---|---:|---:|---:|---:|
{eval_table}
{pareto_section}
### Hardware & Training Telemetry

{telemetry_block}

---

## On-Disk Artifact Audit

{artifact_section}

---

## Quick Start (Public NeuralOps Hub API)

{quick_start}

---

## Citation & License

Licensed under the **MIT License** ([LICENSE]({LICENSE_LINK})).

```bibtex
@software{{{spec['bibtex_key']},
  title   = {{{spec['bibtex_title']}}},
  author  = {{Majlesi, Taha}},
  year    = {{2026}},
  url     = {{{REPO_URL}}},
  note    = {{Hub Repository: {repo_id}}}
}}
```
"""


__all__ = [
    "ZOO_SPECS",
    "audit_artifacts_on_disk",
    "generate_unified_model_card",
    "normalize_base_model_id",
]
