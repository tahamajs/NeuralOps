#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Offline Teacher Representation Caching for Causal LMs & VLMs.

Canonical knowledge distillation cache module in NeuralOps.
Eliminates live teacher GPU memory and forward-pass overhead during student
distillation across multi-mechanism zoos (Qwen2.5-0.5B, TinyLlama-1.1B,
LLaVA-Qwen-0.5B) on RTX 3090 (24 GB VRAM) + 23 GB host RAM clusters.

Key guarantees:
  - Cache hits NEVER instantiate or move a teacher model to GPU (0 MB teacher VRAM).
  - Strided decoder hidden states (FP16) + top-K sparse logits (FP16 vals / INT32 idx)
    bound cache size to ~1.4-1.7 GB (VLM strictly < 1.8 GB).
  - Advisory file locking + atomic temp-file os.replace prevent subprocess race corruption.
  - Automatic batch-size OOM backoff during cache generation.
"""
from __future__ import annotations

import contextlib
import fcntl
import gc
import logging
import math
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("neuralops.benchmarks.kd.teacher_cache")

_env_cache_root = os.environ.get("NEURALOPS_CACHE_DIR")
if _env_cache_root:
    HPC_DATA_CACHE_ROOT = Path(_env_cache_root)
elif Path("/home/tahamajlesi.ut/NeuralOps/data_cache").exists():
    HPC_DATA_CACHE_ROOT = Path("/home/tahamajlesi.ut/NeuralOps/data_cache")
else:
    HPC_DATA_CACHE_ROOT = Path.home() / ".cache" / "neuralops" / "data_cache"

VLM_MAX_CACHE_BYTES = int(1.75 * 1024 * 1024 * 1024)


def _safe_torch_load(path: Path) -> Any:
    """Load a torch checkpoint across PyTorch 1.12 and 2.6+ without weights_only mismatch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(payload: Any, target_path: Path) -> None:
    """Atomically write a torch payload via a unique sibling temp file and os.replace."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f".{target_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as exc:
                logger.warning("Failed to remove temp cache file %s: %s", tmp_path, exc)


@contextlib.contextmanager
def _cache_file_lock(cache_path: Path) -> Iterator[None]:
    """Advisory exclusive file lock around cache inspection and creation."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_name(f"{cache_path.name}.lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def resolve_teacher_cache_path(
    teacher_cache_path: Optional[str],
    out_dir: str | Path,
    model_id: str,
    tag: str,
    seq_len: int,
    max_cache_samples: int,
    topk: int,
) -> Optional[Path]:
    """
    Resolve offline teacher cache path from CLI flag and environment.

    - 'none' / 'off' / 'false' / '0' disables caching (returns None).
    - Explicit path is returned as Path(teacher_cache_path).
    - 'auto' / None uses $NEURALOPS_CACHE_DIR/teacher_caches/<slug>.pt
      or /home/tahamajlesi.ut/NeuralOps/data_cache/teacher_caches/<slug>.pt
      when HPC data_cache exists, otherwise <out_dir>/_teacher_cache.pt.
    """
    if teacher_cache_path is not None:
        cleaned = str(teacher_cache_path).strip()
        if cleaned.lower() in ("none", "off", "false", "0"):
            return None
        if cleaned.lower() not in ("", "auto"):
            return Path(cleaned)

    safe_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id.strip("/")).strip("_") or "model"
    filename = f"{safe_id}_{tag}_seq{int(seq_len)}_n{int(max_cache_samples)}_k{int(topk)}.pt"
    if HPC_DATA_CACHE_ROOT.exists():
        return HPC_DATA_CACHE_ROOT / "teacher_caches" / filename
    return Path(out_dir) / "_teacher_cache.pt"


def has_sufficient_space(target_path: Path, required_bytes: int, safety_margin_bytes: int = 256 * 1024 * 1024) -> bool:
    """Check whether the filesystem containing target_path has enough free space."""
    probe = target_path.parent if target_path.parent.exists() else Path("/tmp")
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    return usage.free >= (required_bytes + safety_margin_bytes)


def select_strided_layer_indices(num_layers: int, max_layers: Optional[int] = None) -> list[int]:
    """
    Select strided decoder layer indices including the final layer.

    For 24 layers without a cap: [3, 7, 11, 15, 19, 23].
    For 22 layers without a cap: [3, 7, 11, 15, 19, 21].
    """
    if num_layers <= 0:
        return []
    indices = sorted(set(list(range(3, num_layers, 4)) + [num_layers - 1]))
    if max_layers is not None and max_layers >= 1 and len(indices) > max_layers:
        stride = max(1, num_layers // max_layers)
        capped = list(range(stride - 1, num_layers, stride))
        if (num_layers - 1) not in capped:
            if len(capped) >= max_layers:
                capped[-1] = num_layers - 1
            else:
                capped.append(num_layers - 1)
        indices = sorted(set(capped[:max_layers]))
        if (num_layers - 1) not in indices:
            indices[-1] = num_layers - 1
    return indices


def select_vlm_layer_indices(
    num_layers: int,
    num_samples: int,
    avg_seq_len: int,
    hidden_size: int,
    max_bytes: int = VLM_MAX_CACHE_BYTES,
) -> list[int]:
    """Choose strided decoder layers so VLM FP16 hidden-state cache stays strictly under max_bytes."""
    indices = select_strided_layer_indices(num_layers)
    if not indices or num_samples <= 0 or avg_seq_len <= 0 or hidden_size <= 0:
        return indices
    bytes_per_layer = num_samples * avg_seq_len * hidden_size * 2
    est_bytes = len(indices) * bytes_per_layer
    if est_bytes <= max_bytes:
        return indices
    max_layers = max(1, int(max_bytes // max(bytes_per_layer, 1)))
    return select_strided_layer_indices(num_layers, max_layers=max_layers)


def _infer_num_layers(model: nn.Module) -> int:
    """Infer decoder layer count across causal LM and multimodal VLM architectures."""
    lm = getattr(model, "language_model", model)
    layers = getattr(getattr(lm, "model", lm), "layers", None)
    if layers is not None:
        return len(layers)
    cfg = getattr(lm, "config", getattr(model, "config", None))
    text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
    num_layers = getattr(text_cfg, "num_hidden_layers", None) if text_cfg is not None else None
    if num_layers is not None:
        return int(num_layers)
    raise ValueError("Unable to determine num_hidden_layers from teacher model.")


def enable_student_gradient_checkpointing(student: nn.Module, log_prefix: str = "") -> None:
    """Enable gradient checkpointing safely on causal LM or VLM language_model."""
    target = getattr(student, "language_model", student)
    if not hasattr(target, "gradient_checkpointing_enable"):
        return
    try:
        if hasattr(student, "config"):
            student.config.use_cache = False
        if hasattr(target, "config"):
            target.config.use_cache = False
        try:
            target.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            target.gradient_checkpointing_enable()
        if hasattr(target, "enable_input_require_grads"):
            target.enable_input_require_grads()
        logger.info("%sGradient checkpointing enabled on student decoder blocks.", log_prefix)
    except Exception as exc:
        logger.warning("%sgradient_checkpointing_enable failed (%s); continuing without.", log_prefix, exc)


def compute_topk_sparse_kd_loss_from_cache(
    student_logits: torch.Tensor,
    t_topk_vals: torch.Tensor,
    t_topk_idx: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Compute top-K sparse logit KL divergence using cached teacher top-K values & indices."""
    tau = float(temperature)
    idx = t_topk_idx.long()
    s_topk_vals = torch.gather(student_logits.float(), dim=-1, index=idx)
    p_s = F.log_softmax(s_topk_vals / tau, dim=-1)
    p_t = F.softmax(t_topk_vals.float() / tau, dim=-1)
    return F.kl_div(p_s, p_t, reduction="batchmean") * (tau * tau)


class CachedCausalDataset(Dataset):
    """CPU-backed dataset yielding token inputs and offline cached teacher targets."""

    def __init__(self, cache: dict[str, Any]) -> None:
        self.input_ids: torch.Tensor = cache["input_ids"]
        self.attention_mask: torch.Tensor = cache["attention_mask"]
        self.hidden_states: torch.Tensor = cache["hidden_states"]
        self.topk_vals: torch.Tensor = cache["topk_vals"]
        self.topk_idx: torch.Tensor = cache["topk_idx"]

    def __len__(self) -> int:
        return int(self.input_ids.size(0))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.input_ids[idx].long()
        return {
            "input_ids": ids,
            "attention_mask": self.attention_mask[idx].long(),
            "labels": ids.clone(),
            "teacher_hidden_states": self.hidden_states[idx],
            "teacher_topk_vals": self.topk_vals[idx],
            "teacher_topk_idx": self.topk_idx[idx],
        }


def _is_valid_causal_cache(
    payload: Any,
    *,
    model_id: str,
    seq_len: int,
    num_samples: int,
    topk: int,
) -> bool:
    """Validate structure and metadata of a loaded causal LM teacher cache."""
    if not isinstance(payload, dict):
        return False
    if payload.get("cache_type") != "causal_lm":
        return False
    if str(payload.get("model_id", "")) != str(model_id):
        return False
    if int(payload.get("seq_len", -1)) != int(seq_len):
        return False
    if int(payload.get("num_samples", -1)) != int(num_samples):
        return False
    if int(payload.get("requested_topk", payload.get("topk", -1))) != int(topk):
        return False
    required = ("input_ids", "attention_mask", "hidden_states", "topk_vals", "topk_idx", "layer_indices", "teacher_ppl")
    if any(k not in payload for k in required):
        return False
    if not isinstance(payload["input_ids"], torch.Tensor) or payload["input_ids"].size(0) != num_samples:
        return False
    if not isinstance(payload["hidden_states"], torch.Tensor) or payload["hidden_states"].size(0) != num_samples:
        return False
    return True


def try_load_causal_teacher_cache(
    cache_path: Optional[Path],
    *,
    model_id: str,
    seq_len: int,
    num_samples: int,
    topk: int,
) -> Optional[dict[str, Any]]:
    """Return a validated causal teacher cache from disk if present, else None (0 GPU work)."""
    if cache_path is None or not cache_path.exists():
        return None
    try:
        loaded = _safe_torch_load(cache_path)
    except Exception as exc:
        logger.warning("Existing causal teacher cache %s unreadable (%s); will rebuild.", cache_path, exc)
        return None
    if _is_valid_causal_cache(
        loaded,
        model_id=model_id,
        seq_len=seq_len,
        num_samples=num_samples,
        topk=topk,
    ):
        logger.info(
            "[TEACHER CACHE HIT] Loaded %s (N=%d, layers=%s, topk=%d, teacher_ppl=%.2f, 0 MB teacher VRAM)",
            cache_path,
            num_samples,
            loaded["layer_indices"],
            loaded["topk"],
            float(loaded["teacher_ppl"]),
        )
        return loaded
    return None


def build_or_load_causal_teacher_cache(
    cache_path: Path,
    *,
    model_id: str,
    normalized_model_id: str,
    train_chunks: torch.Tensor,
    val_chunks: torch.Tensor,
    device: torch.device,
    load_teacher_fn: Callable[[], nn.Module],
    evaluate_ppl_fn: Callable[[nn.Module, DataLoader, torch.device], float],
    seq_len: int = 512,
    max_cache_samples: int = 256,
    topk: int = 64,
    batch_size: int = 4,
    val_batch_size: int = 4,
    teacher_model: Optional[nn.Module] = None,
) -> dict[str, Any]:
    """
    Load an existing causal LM teacher cache or build it once and unload the teacher.

    When a valid cache exists on disk, `load_teacher_fn` is NEVER invoked, ensuring
    0 MB teacher GPU VRAM during student distillation.
    """
    if train_chunks.dim() != 2 or train_chunks.size(0) == 0:
        raise ValueError(f"Expected non-empty 2D train_chunks [N, L], got {tuple(train_chunks.shape)}.")

    n_samples = min(int(train_chunks.size(0)), max(1, int(max_cache_samples)))

    if cache_path.exists():
        try:
            loaded = _safe_torch_load(cache_path)
            if _is_valid_causal_cache(
                loaded,
                model_id=normalized_model_id,
                seq_len=seq_len,
                num_samples=n_samples,
                topk=topk,
            ):
                logger.info(
                    "[TEACHER CACHE HIT] Loaded %s (N=%d, layers=%s, topk=%d, teacher_ppl=%.2f, 0 MB teacher VRAM)",
                    cache_path,
                    n_samples,
                    loaded["layer_indices"],
                    loaded["topk"],
                    float(loaded["teacher_ppl"]),
                )
                return loaded
        except Exception as exc:
            logger.warning("Existing causal teacher cache %s unreadable (%s); re-checking under lock...", cache_path, exc)

    with _cache_file_lock(cache_path):
        if cache_path.exists():
            try:
                loaded = _safe_torch_load(cache_path)
                if _is_valid_causal_cache(
                    loaded,
                    model_id=normalized_model_id,
                    seq_len=seq_len,
                    num_samples=n_samples,
                    topk=topk,
                ):
                    logger.info("[TEACHER CACHE HIT] Loaded %s after lock wait (0 MB teacher VRAM)", cache_path)
                    return loaded
            except Exception as exc:
                logger.warning("Rebuilding stale/corrupt causal cache %s: %s", cache_path, exc)

        owns_teacher = teacher_model is None
        teacher = load_teacher_fn() if owns_teacher else teacher_model
        assert teacher is not None
        try:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False

            class _ValDictDataset(Dataset):
                def __init__(self, chunks: torch.Tensor) -> None:
                    self.chunks = chunks

                def __len__(self) -> int:
                    return int(self.chunks.size(0))

                def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                    ids = self.chunks[idx].long()
                    return {"input_ids": ids, "attention_mask": torch.ones_like(ids, dtype=torch.long), "labels": ids.clone()}

            val_loader = DataLoader(_ValDictDataset(val_chunks), batch_size=max(1, val_batch_size), shuffle=False)
            teacher_ppl = float(evaluate_ppl_fn(teacher, val_loader, device))

            num_layers = _infer_num_layers(teacher)
            layer_indices = select_strided_layer_indices(num_layers)
            sub_chunks = train_chunks[:n_samples]

            cur_bs = max(1, min(int(batch_size), n_samples))
            while True:
                all_ids: list[torch.Tensor] = []
                all_masks: list[torch.Tensor] = []
                all_hiddens: list[torch.Tensor] = []
                all_tvals: list[torch.Tensor] = []
                all_tidx: list[torch.Tensor] = []
                eff_k = int(topk)
                try:
                    with torch.no_grad():
                        for start in range(0, n_samples, cur_bs):
                            end = min(start + cur_bs, n_samples)
                            in_ids = sub_chunks[start:end].to(device=device, dtype=torch.long, non_blocking=True)
                            att_m = torch.ones_like(in_ids, dtype=torch.long)
                            with (torch.cuda.amp.autocast(dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()):
                                out = teacher(
                                    input_ids=in_ids,
                                    attention_mask=att_m,
                                    output_hidden_states=True,
                                    use_cache=False,
                                )
                                raw_h = out.hidden_states
                                h_layers = raw_h[1:] if len(raw_h) == num_layers + 1 else raw_h
                                sel_h = torch.stack([h_layers[idx] for idx in layer_indices], dim=1)
                                logits = out.logits
                                eff_k = min(int(topk), int(logits.size(-1)))
                                t_vals, t_idx = torch.topk(logits, k=eff_k, dim=-1)

                            all_ids.append(in_ids.to(torch.int32).cpu())
                            all_masks.append(att_m.to(torch.uint8).cpu())
                            all_hiddens.append(sel_h.to(torch.float16).cpu())
                            all_tvals.append(t_vals.to(torch.float16).cpu())
                            all_tidx.append(t_idx.to(torch.int32).cpu())
                    break
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and cur_bs > 1:
                        cur_bs = max(1, cur_bs // 2)
                        logger.warning("[TEACHER CACHE] OOM at cache build; backing off to batch_size=%d", cur_bs)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(0.05)
                        continue
                    raise

            payload = {
                "cache_version": 1,
                "cache_type": "causal_lm",
                "model_id": str(normalized_model_id),
                "raw_model_id": str(model_id),
                "seq_len": int(seq_len),
                "num_samples": int(n_samples),
                "requested_max_samples": int(max_cache_samples),
                "num_layers": int(num_layers),
                "layer_indices": [int(i) for i in layer_indices],
                "topk": int(eff_k),
                "requested_topk": int(topk),
                "teacher_ppl": float(teacher_ppl),
                "input_ids": torch.cat(all_ids, dim=0),
                "attention_mask": torch.cat(all_masks, dim=0),
                "hidden_states": torch.cat(all_hiddens, dim=0),
                "topk_vals": torch.cat(all_tvals, dim=0),
                "topk_idx": torch.cat(all_tidx, dim=0),
            }
            atomic_torch_save(payload, cache_path)
            logger.info(
                "[TEACHER CACHE BUILT] Saved %s (N=%d, layers=%s, topk=%d, teacher_ppl=%.2f)",
                cache_path,
                n_samples,
                layer_indices,
                eff_k,
                teacher_ppl,
            )
            return payload
        finally:
            if owns_teacher:
                del teacher
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


def _is_valid_vlm_cache(
    payload: Any,
    *,
    model_id: str,
    num_samples: int,
    topk: int,
) -> bool:
    """Validate structure and metadata of a loaded VLM teacher cache."""
    if not isinstance(payload, dict):
        return False
    if payload.get("cache_type") != "vlm_scienceqa":
        return False
    if str(payload.get("model_id", "")) != str(model_id):
        return False
    if int(payload.get("num_samples", -1)) != int(num_samples):
        return False
    if int(payload.get("requested_topk", payload.get("topk", -1))) != int(topk):
        return False
    required = (
        "hidden_states",
        "tail_topk_vals",
        "tail_topk_idx",
        "tail_start",
        "seq_lens",
        "layer_indices",
        "teacher_acc",
        "teacher_ppl",
    )
    if any(k not in payload for k in required):
        return False
    if len(payload["hidden_states"]) != num_samples:
        return False
    if len(payload["tail_topk_vals"]) != num_samples:
        return False
    return True


def try_load_vlm_teacher_cache(
    cache_path: Optional[Path],
    *,
    model_id: str,
    num_samples: int,
    topk: int,
) -> Optional[dict[str, Any]]:
    """Return a validated VLM teacher cache from disk if present, else None (0 GPU work)."""
    if cache_path is None or not cache_path.exists():
        return None
    try:
        loaded = _safe_torch_load(cache_path)
    except Exception as exc:
        logger.warning("Existing VLM teacher cache %s unreadable (%s); will rebuild.", cache_path, exc)
        return None
    if _is_valid_vlm_cache(loaded, model_id=model_id, num_samples=num_samples, topk=topk):
        logger.info(
            "[VLM TEACHER CACHE HIT] Loaded %s (N=%d, layers=%s, teacher_acc=%.2f%%, teacher_ppl=%.2f, 0 MB teacher VRAM)",
            cache_path,
            num_samples,
            loaded["layer_indices"],
            float(loaded["teacher_acc"]),
            float(loaded["teacher_ppl"]),
        )
        return loaded
    return None


def build_or_load_vlm_teacher_cache(
    cache_path: Path,
    *,
    model_id: str,
    normalized_model_id: str,
    train_examples: list[dict[str, Any]],
    val_examples: list[dict[str, Any]],
    processor: Any,
    device: torch.device,
    load_teacher_fn: Callable[[], nn.Module],
    evaluate_vlm_fn: Callable[..., tuple[float, float]],
    collate_fn: Callable[..., dict[str, torch.Tensor]],
    max_cache_samples: int = 256,
    topk: int = 64,
    tail_tokens: int = 32,
    batch_size: int = 4,
    teacher_model: Optional[nn.Module] = None,
) -> dict[str, Any]:
    """
    Load an existing VLM teacher cache or build it once and unload the teacher.

    References `train_examples` by index so `pixel_values` are NEVER duplicated.
    Automatically bounds strided layer count to keep total VLM cache < 1.8 GB.
    """
    if not train_examples:
        raise ValueError("Cannot build VLM teacher cache with 0 training examples.")

    n_samples = min(len(train_examples), max(1, int(max_cache_samples)))

    if cache_path.exists():
        try:
            loaded = _safe_torch_load(cache_path)
            if _is_valid_vlm_cache(loaded, model_id=normalized_model_id, num_samples=n_samples, topk=topk):
                logger.info(
                    "[VLM TEACHER CACHE HIT] Loaded %s (N=%d, layers=%s, teacher_acc=%.2f%%, teacher_ppl=%.2f, 0 MB teacher VRAM)",
                    cache_path,
                    n_samples,
                    loaded["layer_indices"],
                    float(loaded["teacher_acc"]),
                    float(loaded["teacher_ppl"]),
                )
                return loaded
        except Exception as exc:
            logger.warning("Existing VLM teacher cache %s unreadable (%s); re-checking under lock...", cache_path, exc)

    with _cache_file_lock(cache_path):
        if cache_path.exists():
            try:
                loaded = _safe_torch_load(cache_path)
                if _is_valid_vlm_cache(loaded, model_id=normalized_model_id, num_samples=n_samples, topk=topk):
                    logger.info("[VLM TEACHER CACHE HIT] Loaded %s after lock wait (0 MB teacher VRAM)", cache_path)
                    return loaded
            except Exception as exc:
                logger.warning("Rebuilding stale/corrupt VLM cache %s: %s", cache_path, exc)

        owns_teacher = teacher_model is None
        teacher = load_teacher_fn() if owns_teacher else teacher_model
        assert teacher is not None
        try:
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False

            eval_bs = max(1, int(batch_size))
            while True:
                try:
                    teacher_acc, teacher_ppl = evaluate_vlm_fn(teacher, val_examples, processor, device, batch_size=eval_bs)
                    break
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and eval_bs > 1:
                        eval_bs = max(1, eval_bs // 2)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(0.05)
                        continue
                    raise

            num_layers = _infer_num_layers(teacher)
            cfg = getattr(getattr(teacher, "language_model", teacher), "config", getattr(teacher, "config", None))
            text_cfg = getattr(cfg, "text_config", cfg) if cfg is not None else None
            hidden_size = int(getattr(text_cfg, "hidden_size", 1024))
            sub_examples = train_examples[:n_samples]
            avg_len = max(1, int(sum(int(ex["input_ids"].size(0)) for ex in sub_examples) / n_samples))
            layer_indices = select_vlm_layer_indices(num_layers, n_samples, avg_len, hidden_size)

            pad_id = getattr(getattr(processor, "tokenizer", None), "pad_token_id", None)
            if pad_id is None:
                pad_id = 0

            cur_bs = max(1, min(int(batch_size), n_samples))
            while True:
                cached_hiddens: list[torch.Tensor] = []
                cached_tvals: list[torch.Tensor] = []
                cached_tidx: list[torch.Tensor] = []
                cached_tail_starts: list[int] = []
                cached_seq_lens: list[int] = []
                cached_input_ids: list[torch.Tensor] = []
                eff_k = int(topk)
                try:
                    with torch.no_grad():
                        for start in range(0, n_samples, cur_bs):
                            end = min(start + cur_bs, n_samples)
                            b_items = sub_examples[start:end]
                            batch = collate_fn(b_items, pad_id=pad_id, eval_prompt_only=False)
                            in_ids = batch["input_ids"].to(device, non_blocking=True)
                            att_m = batch["attention_mask"].to(device, non_blocking=True)
                            px = batch["pixel_values"].to(
                                device,
                                dtype=torch.float16 if device.type == "cuda" else torch.float32,
                                non_blocking=True,
                            )
                            with (torch.cuda.amp.autocast(dtype=torch.float16) if device.type == "cuda" else contextlib.nullcontext()):
                                out = teacher(
                                    input_ids=in_ids,
                                    attention_mask=att_m,
                                    pixel_values=px,
                                    output_hidden_states=True,
                                    use_cache=False,
                                )
                                raw_h = out.hidden_states
                                h_layers = raw_h[1:] if len(raw_h) == num_layers + 1 else raw_h
                                sel_h = torch.stack([h_layers[idx] for idx in layer_indices], dim=1)
                                logits = out.logits
                                eff_k = min(int(topk), int(logits.size(-1)))

                            for b_i, ex in enumerate(b_items):
                                cur_len = int(ex["input_ids"].size(0))
                                h_ex = sel_h[b_i, :, :cur_len, :].to(torch.float16).cpu().contiguous()
                                t_len = min(int(tail_tokens), cur_len)
                                t_start = cur_len - t_len
                                tail_logits = logits[b_i, t_start:cur_len, :]
                                t_vals, t_idx = torch.topk(tail_logits, k=eff_k, dim=-1)
                                cached_hiddens.append(h_ex)
                                cached_tvals.append(t_vals.to(torch.float16).cpu().contiguous())
                                cached_tidx.append(t_idx.to(torch.int32).cpu().contiguous())
                                cached_tail_starts.append(int(t_start))
                                cached_seq_lens.append(int(cur_len))
                                cached_input_ids.append(ex["input_ids"].to(torch.int32).cpu())
                    break
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower() and cur_bs > 1:
                        cur_bs = max(1, cur_bs // 2)
                        logger.warning("[VLM TEACHER CACHE] OOM at cache build; backing off to batch_size=%d", cur_bs)
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect()
                        time.sleep(0.05)
                        continue
                    raise

            payload = {
                "cache_version": 1,
                "cache_type": "vlm_scienceqa",
                "model_id": str(normalized_model_id),
                "raw_model_id": str(model_id),
                "num_samples": int(n_samples),
                "requested_max_samples": int(max_cache_samples),
                "example_indices": list(range(n_samples)),
                "num_layers": int(num_layers),
                "layer_indices": [int(i) for i in layer_indices],
                "topk": int(eff_k),
                "requested_topk": int(topk),
                "tail_tokens": int(tail_tokens),
                "teacher_acc": float(teacher_acc),
                "teacher_ppl": float(teacher_ppl) if math.isfinite(teacher_ppl) else float("inf"),
                "input_ids": cached_input_ids,
                "hidden_states": cached_hiddens,
                "tail_topk_vals": cached_tvals,
                "tail_topk_idx": cached_tidx,
                "tail_start": cached_tail_starts,
                "seq_lens": cached_seq_lens,
            }
            atomic_torch_save(payload, cache_path)
            logger.info(
                "[VLM TEACHER CACHE BUILT] Saved %s (N=%d, layers=%s, topk=%d, acc=%.2f%%, ppl=%.2f)",
                cache_path,
                n_samples,
                layer_indices,
                eff_k,
                teacher_acc,
                teacher_ppl,
            )
            return payload
        finally:
            if owns_teacher:
                del teacher
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()


__all__ = [
    "HPC_DATA_CACHE_ROOT",
    "VLM_MAX_CACHE_BYTES",
    "atomic_torch_save",
    "resolve_teacher_cache_path",
    "has_sufficient_space",
    "select_strided_layer_indices",
    "select_vlm_layer_indices",
    "enable_student_gradient_checkpointing",
    "compute_topk_sparse_kd_loss_from_cache",
    "CachedCausalDataset",
    "try_load_causal_teacher_cache",
    "build_or_load_causal_teacher_cache",
    "try_load_vlm_teacher_cache",
    "build_or_load_vlm_teacher_cache",
]
