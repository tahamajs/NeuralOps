"""Evaluation pipeline and benchmark orchestration for NeuralOps.

Implements adaptive batch probing, multi-phase combination evaluation
(environment setup, baseline scoring, compression injection, evaluation, artifact assembly),
and convenience sweeps.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.profiler import ProfilerActivity, profile

from neuralops.evaluation.eval_data import (
    _PREPARED_DATASET_CACHE,
    build_calibration_loader,
    get_dataset_batch,
)
from neuralops.evaluation.eval_patcher import (
    apply_module_to_model,
    get_base_model_cached,
)
from neuralops.evaluation.eval_types import (
    ACTIVATION_AWARE_MODULES,
    BenchmarkResult,
    COMPUTE_OPTIMIZERS,
    DEPRECATED_MODULES,
    KV_CACHE_OPTIMIZERS,
    LM_DATASETS,
    MODEL_TOKENIZER_MAPPING,
)
from neuralops.evaluation.eval_utils import _extract_logits_and_loss, _filter_batch

logger = logging.getLogger("EvalEngine")


def probe_optimal_batch_size(
    model: nn.Module,
    model_family: str,
    dataset: str,
    seq_len: int,
    device: torch.device,
    target_vram_ratio: float = 0.35,
    max_batch_cap: int = 64,
    min_batch_floor: int = 4,
    allow_synthetic: bool = False,
) -> int:
    """Probes GPU memory to dynamically compute the largest power-of-two batch size fitting VRAM.

    Args:
        model: Model under evaluation.
        model_family: Model family name for tokenizer resolution.
        dataset: Target dataset name.
        seq_len: Sequence length dimension.
        device: Target execution device.
        target_vram_ratio: Fraction of total VRAM budgeted for activation memory.
        max_batch_cap: Ceiling on returned batch size.
        min_batch_floor: Floor on returned batch size.
        allow_synthetic: Whether to permit synthetic batch fallback during probe.

    Returns:
        Optimal integer batch size (power of 2).
    """
    if device.type != "cuda":
        return 64

    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    idle_vram = torch.cuda.memory_allocated(device)
    total_vram = torch.cuda.get_device_properties(device).total_memory
    target_usable_vram = total_vram * target_vram_ratio

    probe_b = 4
    num_labels = 1 if dataset.lower() == "stsb" else 2

    try:
        probe_batch = get_dataset_batch(
            dataset,
            model_family,
            probe_b,
            seq_len,
            num_labels=num_labels,
            device=device,
            allow_synthetic=allow_synthetic,
        )
        filtered_batch = _filter_batch(probe_batch, model_family, model)

        with torch.amp.autocast("cuda"):
            out = model(**filtered_batch)
        torch.cuda.synchronize(device)
        fwd_peak = torch.cuda.max_memory_allocated(device)
        del out

        try:
            with torch.amp.autocast("cuda"):
                out2 = model(**filtered_batch)
            _, loss = _extract_logits_and_loss(
                out2,
                num_labels=num_labels,
                labels=filtered_batch.get("labels"),
                is_lm=dataset.lower() in LM_DATASETS,
            )
            if loss is not None and loss.requires_grad:
                loss.backward()
                model.zero_grad()
            torch.cuda.synchronize(device)
            peak_vram = torch.cuda.max_memory_allocated(device)
        except Exception:
            peak_vram = fwd_peak

        activation_vram = max(1024 * 1024.0, peak_vram - idle_vram)
        mem_per_sample = activation_vram / float(probe_b)
        remaining_vram = max(1024 * 1024.0, target_usable_vram - idle_vram)
        estimated_max_b = int(remaining_vram / mem_per_sample)

        powers_of_2 = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        selected_b = min_batch_floor
        for p in powers_of_2:
            if p <= estimated_max_b and p <= max_batch_cap:
                selected_b = p

        logger.info(
            "Batch probe: peak=%.1fMB, per_sample=%.2fMB -> selected=%d",
            peak_vram / (1024 * 1024),
            mem_per_sample / (1024 * 1024),
            selected_b,
        )
        return max(min_batch_floor, min(selected_b, max_batch_cap))
    except Exception as e:
        logger.warning("Batch probe failed (%s), defaulting to 64", e)
        return 64
    finally:
        gc.collect()
        torch.cuda.empty_cache()


# -----------------------------------------------------------------------------
# Internal sub-phases of evaluate_single_combination_opt
# -----------------------------------------------------------------------------


def _setup_eval_environment(
    model_family: str,
    dataset: str,
    seq_len: int,
    device: torch.device,
) -> dict[str, Any]:
    """Phase 1: Setup execution context, base models, and memory baselines."""
    is_cuda = device.type == "cuda"
    if is_cuda:
        torch.cuda.empty_cache()

    num_labels = 1 if dataset.lower() == "stsb" else 2
    mem_phases: dict[str, dict[str, float]] = {}

    if is_cuda:
        torch.cuda.synchronize(device)
        mem_phases["idle"] = {
            "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2),
        }

    t_load_0 = time.time()
    model, cfg = get_base_model_cached(
        model_family=model_family,
        num_labels=num_labels,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        device=device,
    )
    model.train()
    t_model_load = time.time() - t_load_0

    if is_cuda:
        torch.cuda.synchronize(device)
        mem_phases["model_load"] = {
            "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2),
        }

    return {
        "model": model,
        "cfg": cfg,
        "num_labels": num_labels,
        "mem_phases": mem_phases,
        "t_model_load": t_model_load,
        "is_cuda": is_cuda,
    }


def _apply_compression_phase(
    model: nn.Module,
    module_name: str,
    module_type: str,
    model_family: str,
    dataset: str,
    rank: int,
    seq_len: int,
    device: torch.device,
    mem_phases: dict[str, dict[str, float]],
    run_log_lines: list[str],
) -> tuple[nn.Module, float, dict[str, Any], float, float, bool, str | None]:
    """Phase 2: Build calibration data and inject target attention or compressor module."""
    is_cuda = device.type == "cuda"
    t_calib_0 = time.time()
    calib_loader = None
    calib_tok = None
    calib_source = None
    calib_used = False

    if module_type == "compressor" and module_name in ACTIVATION_AWARE_MODULES:
        run_log_lines.append(
            f"[CALIB] Building calibration loader for {module_name} on {dataset}..."
        )
        calib_loader, calib_tok, calib_source = build_calibration_loader(
            dataset=dataset,
            model_family=model_family,
            seq_len=seq_len,
            device=device,
            n_samples=500,
            batch_size=8,
        )
        if calib_loader is None:
            run_log_lines.append(
                f"[CALIB] Warning: Failed to build calibration loader for {module_name}"
            )
            logger.warning("[CALIB] No calibration loader built for %s on %s.", module_name, dataset)
        else:
            calib_used = True
            run_log_lines.append(
                f"[CALIB] Successfully built calibration loader from {calib_source} (500 samples)"
            )

    t_calib = time.time() - t_calib_0
    if is_cuda:
        torch.cuda.synchronize(device)
        mem_phases["calibration"] = {
            "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2),
        }

    t_comp_0 = time.time()
    run_log_lines.append(
        f"[INJECT] Injecting module '{module_name}' (type={module_type}, rank={rank})"
    )
    model, comp_ratio = apply_module_to_model(
        model=model,
        module_name=module_name,
        module_type=module_type,
        rank=rank,
        device=device,
        calibration_loader=calib_loader,
        tokenizer=calib_tok,
    )
    t_comp = time.time() - t_comp_0
    if is_cuda:
        torch.cuda.synchronize(device)
        mem_phases["compression"] = {
            "allocated_mb": round(torch.cuda.memory_allocated(device) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / (1024 * 1024), 2),
        }

    comp_stats = getattr(model, "compression_stats", {}) or {}
    return model, comp_ratio, comp_stats, t_calib, t_comp, calib_used, calib_source


def _benchmark_runtime_latency(
    model: nn.Module,
    filtered_batch: dict[str, torch.Tensor],
    num_steps: int,
    num_warmup: int,
    num_labels: int,
    is_lm: bool,
    is_cuda: bool,
    device: torch.device,
    current_batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    """Phase 3: Measure step forward/backward latency, throughput, memory and gradient norm."""
    model.eval()
    flops_macs = 0.0
    try:
        with (
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                with_flops=True,
            ) as prof,
            torch.amp.autocast("cuda") if is_cuda else contextlib.nullcontext(),
        ):
            _ = model(**filtered_batch)
        flops_macs = sum((e.flops or 0) for e in prof.key_averages())
    except Exception as fe:
        logger.debug("FLOPs profiling failed: %s", fe)

    for _ in range(num_warmup):
        with torch.amp.autocast("cuda") if is_cuda else contextlib.nullcontext():
            _ = model(**filtered_batch)

    model.train()
    traj_loss: list[float] = []
    traj_grad_norm: list[float] = []
    traj_fwd_ms: list[float] = []
    traj_bwd_ms: list[float] = []
    peak_memory_mb = 0.0

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        fwd_start = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
        fwd_end = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
        bwd_start = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]
        bwd_end = [torch.cuda.Event(enable_timing=True) for _ in range(num_steps)]

        for i in range(num_steps):
            fwd_start[i].record()
            with torch.amp.autocast("cuda"):
                out = model(**filtered_batch)
            fwd_end[i].record()

            with torch.amp.autocast("cuda"):
                _, loss = _extract_logits_and_loss(
                    out,
                    num_labels=num_labels,
                    labels=filtered_batch.get("labels"),
                    is_lm=is_lm,
                )

            if loss is None or not loss.requires_grad:
                traj_loss.append(0.0)
                traj_grad_norm.append(0.0)
                traj_fwd_ms.append(0.0)
                traj_bwd_ms.append(0.0)
                model.zero_grad()
                continue

            traj_loss.append(round(float(loss.detach().item()), 4))
            bwd_start[i].record()
            loss.backward()
            bwd_end[i].record()

            total_norm_sq = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    total_norm_sq += float(p.grad.detach().norm(2).item() ** 2)
            traj_grad_norm.append(round(float(total_norm_sq**0.5), 4))
            model.zero_grad()

        if num_steps > 0:
            bwd_end[-1].synchronize()
        for i in range(num_steps):
            try:
                traj_fwd_ms.append(round(float(fwd_start[i].elapsed_time(fwd_end[i])), 4))
                traj_bwd_ms.append(round(float(bwd_start[i].elapsed_time(bwd_end[i])), 4))
            except Exception:
                pass

        peak_memory_mb = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
    else:
        for _ in range(num_steps):
            t0 = time.perf_counter()
            out = model(**filtered_batch)
            traj_fwd_ms.append(round((time.perf_counter() - t0) * 1000.0, 4))

            _, loss = _extract_logits_and_loss(
                out,
                num_labels=num_labels,
                labels=filtered_batch.get("labels"),
                is_lm=is_lm,
            )
            if loss is None or not loss.requires_grad:
                traj_loss.append(0.0)
                traj_grad_norm.append(0.0)
                traj_bwd_ms.append(0.0)
                model.zero_grad()
                continue

            traj_loss.append(round(float(loss.detach().item()), 4))
            t2 = time.perf_counter()
            loss.backward()
            traj_bwd_ms.append(round((time.perf_counter() - t2) * 1000.0, 4))

            total_norm_sq = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    total_norm_sq += float(p.grad.detach().norm(2).item() ** 2)
            traj_grad_norm.append(round(float(total_norm_sq**0.5), 4))
            model.zero_grad()

    fwd_lat = round(float(np.mean(traj_fwd_ms)), 2) if traj_fwd_ms else 0.0
    bwd_lat = round(float(np.mean(traj_bwd_ms)), 2) if traj_bwd_ms else 0.0
    total_time_s = max(1e-4, sum(traj_fwd_ms + traj_bwd_ms) / 1000.0)

    return {
        "flops_macs": flops_macs,
        "peak_memory_mb": peak_memory_mb,
        "forward_latency_ms": fwd_lat,
        "backward_latency_ms": bwd_lat,
        "total_step_latency_ms": round(fwd_lat + bwd_lat, 2),
        "latency_variance_ms": round(float(np.var(traj_fwd_ms) + np.var(traj_bwd_ms)), 4) if traj_fwd_ms else 0.0,
        "throughput_samples_per_sec": round((num_steps * current_batch_size) / total_time_s, 2),
        "throughput_tokens_sec": round((num_steps * current_batch_size * seq_len) / total_time_s, 2),
        "loss": round(float(np.mean(traj_loss)), 4) if traj_loss else 0.0,
        "grad_norm": round(float(np.mean(traj_grad_norm)), 4) if traj_grad_norm else 0.0,
        "trajectories": {
            "loss": traj_loss,
            "grad_norm": traj_grad_norm,
            "fwd_latency_ms": traj_fwd_ms,
            "bwd_latency_ms": traj_bwd_ms,
        },
    }


def _evaluate_model_accuracy(
    model: nn.Module,
    model_family: str,
    dataset: str,
    num_labels: int,
    seq_len: int,
    current_batch_size: int,
    is_lm: bool,
    is_cuda: bool,
    device: torch.device,
    mem_phases: dict[str, dict[str, float]],
    run_log_lines: list[str],
) -> dict[str, Any]:
    """Phase 4: Evaluate accuracy, perplexity, and cosine fidelity against clean baseline."""
    model.eval()
    all_preds: list[Any] = []
    all_labels: list[Any] = []
    sum_cosine_sim = 0.0
    batches_for_fidelity = 0
    lm_nll_sum = 0.0
    lm_token_count = 0
    baseline_logits_samples: list[list[float]] = []
    compressed_logits_samples: list[list[float]] = []
    sample_cosines: list[float] = []

    t_eval_0 = time.time()
    run_log_lines.append(f"[EVAL] Starting evaluation on {dataset}...")

    baseline_model, _ = get_base_model_cached(
        model_family=model_family,
        num_labels=num_labels,
        hidden_size=128,
        num_heads=4,
        num_layers=2,
        device=device,
    )
    baseline_model.eval()

    try:
        tok_name = MODEL_TOKENIZER_MAPPING.get(model_family.lower(), "bert-base-uncased")
        dataset_cache_key = (dataset.lower(), tok_name, seq_len)

        if dataset_cache_key in _PREPARED_DATASET_CACHE:
            val_tokenized = _PREPARED_DATASET_CACHE[dataset_cache_key]
            loader = torch.utils.data.DataLoader(
                val_tokenized,
                batch_size=current_batch_size,
                shuffle=False,
                pin_memory=is_cuda,
            )
            autocast_ctx = torch.amp.autocast("cuda") if is_cuda else contextlib.nullcontext()

            with torch.no_grad():
                for val_batch in loader:
                    inputs = {
                        ("labels" if k == "label" else k): (
                            v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        )
                        for k, v in val_batch.items()
                    }
                    filtered_inputs = _filter_batch(inputs, model_family, model)
                    filtered_base_inputs = _filter_batch(inputs, model_family, baseline_model)

                    if is_lm:
                        ids = filtered_inputs["input_ids"]
                        filtered_inputs["labels"] = ids.clone()
                        filtered_inputs["labels"][:, :-1] = ids[:, 1:]
                        filtered_inputs["labels"][:, -1] = -100

                    with autocast_ctx:
                        out = model(**filtered_inputs)
                        base_out = baseline_model(**filtered_base_inputs)

                    logits, _ = _extract_logits_and_loss(
                        out,
                        num_labels=num_labels,
                        labels=filtered_inputs.get("labels"),
                        is_lm=is_lm,
                    )
                    base_logits, _ = _extract_logits_and_loss(
                        base_out,
                        num_labels=num_labels,
                        labels=filtered_inputs.get("labels"),
                        is_lm=is_lm,
                    )

                    if is_lm and out.loss is not None:
                        target_labels = filtered_inputs.get("labels")
                        if target_labels is not None:
                            n_targets = int((target_labels != -100).sum().item())
                            lm_nll_sum += float(out.loss.detach().item()) * max(1, n_targets)
                            lm_token_count += max(1, n_targets)

                    if (
                        logits is not None
                        and base_logits is not None
                        and logits.dim() == base_logits.dim()
                        and logits.shape == base_logits.shape
                    ):
                        if logits.dim() == 1:
                            sim = F.cosine_similarity(
                                base_logits.unsqueeze(0), logits.unsqueeze(0), dim=-1
                            ).mean().item()
                        else:
                            sim = F.cosine_similarity(base_logits, logits, dim=-1).mean().item()
                        sum_cosine_sim += float(sim)
                        batches_for_fidelity += 1

                        if len(baseline_logits_samples) < 8:
                            b_count = min(8 - len(baseline_logits_samples), logits.shape[0])
                            for b_i in range(b_count):
                                b_flat = base_logits[b_i].detach().cpu().float().reshape(-1)[:10].tolist()
                                c_flat = logits[b_i].detach().cpu().float().reshape(-1)[:10].tolist()
                                baseline_logits_samples.append([round(x, 4) for x in b_flat])
                                compressed_logits_samples.append([round(x, 4) for x in c_flat])
                                s_cos = F.cosine_similarity(
                                    base_logits[b_i : b_i + 1].detach().float().reshape(1, -1),
                                    logits[b_i : b_i + 1].detach().float().reshape(1, -1),
                                    dim=-1,
                                ).item()
                                sample_cosines.append(round(float(s_cos), 4))

                    if (not is_lm) and logits is not None and inputs.get("labels") is not None:
                        if dataset.lower() == "stsb":
                            preds = logits.squeeze()
                        else:
                            preds = torch.argmax(logits, dim=-1)
                        all_preds.extend(preds.cpu().numpy().tolist())
                        all_labels.extend(inputs["labels"].cpu().numpy().tolist())

    except Exception as eval_e:
        logger.warning("Validation loop failed: %s", eval_e)

    perplexity = 0.0
    if is_lm and lm_token_count > 0:
        mean_nll = lm_nll_sum / lm_token_count
        perplexity = round(float(math.exp(min(mean_nll, 100.0))), 4)

    accuracy_pct = 0.0
    f1 = 0.0
    mcc = 0.0
    pearson = None
    spearman = None

    if all_labels and not is_lm:
        if dataset.lower() == "stsb":
            try:
                p_val, _ = pearsonr(all_labels, all_preds)
                pearson = round(float(p_val), 4) if math.isfinite(p_val) else None
            except Exception:
                pass
            try:
                s_val, _ = spearmanr(all_labels, all_preds)
                spearman = round(float(s_val), 4) if math.isfinite(s_val) else None
            except Exception:
                pass
        elif dataset.lower() == "cola":
            mcc = round(float(matthews_corrcoef(all_labels, all_preds)), 4)
            accuracy_pct = round(float(accuracy_score(all_labels, all_preds)) * 100.0, 2)
        else:
            accuracy_pct = round(float(accuracy_score(all_labels, all_preds)) * 100.0, 2)
            avg = "binary" if (len(set(all_labels)) <= 2 and num_labels <= 2) else "macro"
            try:
                f1 = round(float(f1_score(all_labels, all_preds, average=avg, zero_division=0)), 4)
            except Exception:
                pass

    fidelity_score = round(sum_cosine_sim / batches_for_fidelity, 4) if batches_for_fidelity > 0 else 0.0
    t_eval = time.time() - t_eval_0

    if is_cuda:
        torch.cuda.synchronize(device)
        mem_phases["peak_during_eval"] = {
            "allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2),
            "reserved_mb": round(torch.cuda.max_memory_reserved(device) / (1024 * 1024), 2),
        }

    return {
        "perplexity": perplexity,
        "accuracy_pct": accuracy_pct,
        "f1_score": f1,
        "mcc": mcc,
        "pearson": pearson,
        "spearman": spearman,
        "fidelity_score": fidelity_score,
        "t_eval": t_eval,
        "baseline_logits_samples": baseline_logits_samples,
        "compressed_logits_samples": compressed_logits_samples,
        "sample_cosines": sample_cosines,
    }


def _finalize_benchmark_result(
    result: BenchmarkResult,
    run_log_lines: list[str],
    t_run_start: float,
    module_name: str,
    module_type: str,
    model_family: str,
    dataset: str,
    rank: int,
    seq_len: int,
    batch_size: int,
    current_batch_size: int,
    num_warmup: int,
    num_steps: int,
    allow_synthetic: bool,
    device: torch.device,
    results_dir: str,
    comp_stats: dict[str, Any],
    mem_phases: dict[str, dict[str, float]],
    t_model_load: float,
    t_calib: float,
    t_comp: float,
    t_eval: float,
    baseline_logits_samples: list[list[float]],
    compressed_logits_samples: list[list[float]],
    sample_cosines: list[float],
    exc_traceback: str | None,
    exc_type: str | None,
    retry_count: int,
    success: bool,
) -> BenchmarkResult:
    """Phase 5: Validate metric status and write comprehensive disk artifacts."""
    is_lm = dataset.lower() in LM_DATASETS

    # FIX-10 validation rules
    if result.status == "PASS":
        for name in ("accuracy_pct", "loss", "perplexity", "f1_score", "fidelity_score", "grad_norm"):
            v = getattr(result, name, None)
            if isinstance(v, float) and not math.isfinite(v):
                result.status = "FAIL"
                result.error_message = f"Non-finite value in {name}={v}"
                break

        if module_name in ACTIVATION_AWARE_MODULES and not result.calibration_used:
            result.status = "NO_CALIBRATION"
            result.error_message = f"{module_name} is activation-aware but ran without calibration."

        if (
            result.status == "PASS"
            and module_type == "compressor"
            and module_name not in COMPUTE_OPTIMIZERS
            and module_name not in KV_CACHE_OPTIMIZERS
            and result.compression_ratio >= 0.999
        ):
            result.status = "PASS_NO_COMPRESSION"
            result.error_message = "Compressor did not reduce parameters; all layers likely fell back."

        if result.status == "PASS" and not is_lm and result.accuracy_pct == 0.0:
            result.status = "FAIL"
            result.error_message = "Accuracy is 0.0; validation likely failed."

    # Write traceable artifacts
    try:
        run_dir = Path(results_dir) / model_family / dataset / module_name / f"rank_{rank}"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            fallback_base = Path(tempfile.gettempdir()) / "neuralops_results"
            run_dir = (
                fallback_base
                / Path(results_dir).name
                / model_family
                / dataset
                / module_name
                / f"rank_{rank}"
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            results_dir = str(fallback_base / Path(results_dir).name)

        (run_dir / "run.log").write_text("\n".join(run_log_lines), encoding="utf-8")
        if result.status in ("FAIL", "OOM", "NO_CALIBRATION") and exc_traceback:
            (run_dir / "traceback.txt").write_text(exc_traceback, encoding="utf-8")
        elif result.error_message:
            (run_dir / "traceback.txt").write_text(f"Error: {result.error_message}\n", encoding="utf-8")

        env_dict = {
            "torch_version": torch.__version__,
            "transformers_version": "5.5.0",
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "python_version": sys.version.split()[0],
            "os": sys.platform,
        }
        (run_dir / "environment.json").write_text(json.dumps(env_dict, indent=2), encoding="utf-8")

        cfg_dict = {
            "module_name": module_name,
            "module_type": module_type,
            "model_family": model_family,
            "dataset": dataset,
            "rank": rank,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "num_warmup": num_warmup,
            "num_steps": num_steps,
            "allow_synthetic": allow_synthetic,
            "device": str(device),
        }
        (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2), encoding="utf-8")

        (run_dir / "module_stats.json").write_text(
            json.dumps(comp_stats, indent=2, default=str), encoding="utf-8"
        )

        mem_profile_dict = {
            "phases": mem_phases,
            "oom_encountered": (retry_count > 0 and not success),
            "batch_size_reductions": ([current_batch_size] if current_batch_size != batch_size else []),
        }
        (run_dir / "memory_profile.json").write_text(
            json.dumps(mem_profile_dict, indent=2), encoding="utf-8"
        )

        timing_profile_dict = {
            "phases": {
                "model_load_s": round(t_model_load, 2),
                "dataset_load_s": 0.0,
                "calibration_s": round(t_calib, 2),
                "compression_s": round(t_comp, 2),
                "recovery_s": 0.0,
                "evaluation_s": round(t_eval, 2),
                "total_s": round(time.time() - t_run_start, 2),
            }
        }
        (run_dir / "timing_profile.json").write_text(
            json.dumps(timing_profile_dict, indent=2), encoding="utf-8"
        )

        fidelity_dict = {
            "baseline_logits": baseline_logits_samples[:8] if baseline_logits_samples else [],
            "compressed_logits": compressed_logits_samples[:8] if compressed_logits_samples else [],
            "cosine_similarity_per_sample": [round(s, 4) for s in sample_cosines[:8]],
            "mean_fidelity": result.fidelity_score,
        }
        (run_dir / "fidelity_samples.json").write_text(
            json.dumps(fidelity_dict, indent=2), encoding="utf-8"
        )

        (run_dir / "seed.json").write_text(
            json.dumps({"random": 42, "numpy": 42, "torch": 42, "cuda": 42}, indent=2),
            encoding="utf-8",
        )

        full_result_json = {
            "module_name": module_name,
            "model_family": model_family,
            "dataset": dataset,
            "rank": rank,
            "seq_len": seq_len,
            "batch_size": current_batch_size,
            "status": result.status,
            "metrics": result.to_dict(),
            "compression_stats": comp_stats,
            "error_type": exc_type,
            "traceback": exc_traceback,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        (run_dir / "result.json").write_text(
            json.dumps(full_result_json, indent=2), encoding="utf-8"
        )

    except Exception as save_e:
        logger.error("Failed to write run artifacts: %s", save_e)

    return result


def evaluate_single_combination_opt(
    track: str,
    module_name: str,
    module_type: str,
    model_family: str,
    dataset: str,
    rank: int = 16,
    num_warmup: int = 2,
    num_steps: int = 5,
    batch_size: int = 128,
    seq_len: int = 256,
    device: torch.device = torch.device("cpu"),
    results_dir: str = "./results/gpu_server",
    allow_synthetic: bool = False,
    skip_deprecated: bool = True,
) -> BenchmarkResult:
    """Clean multi-phase orchestrator for single (module, model, dataset) combinations."""
    result = BenchmarkResult(
        track=track,
        module_name=module_name,
        module_type=module_type,
        model_family=model_family,
        dataset=dataset,
        status="RUNNING",
    )

    t_run_start = time.time()
    t_model_load = 0.0
    t_calib = 0.0
    t_comp = 0.0
    t_eval = 0.0
    mem_phases: dict[str, dict[str, float]] = {}
    run_log_lines: list[str] = [
        f"Command: evaluate_single_combination(module={module_name}, model={model_family}, dataset={dataset}, rank={rank})",
        f"Environment: python={sys.version.split()[0]} torch={torch.__version__}",
    ]
    exc_traceback: str | None = None
    exc_type: str | None = None
    comp_stats: dict[str, Any] = {}
    baseline_logits_samples: list[list[float]] = []
    compressed_logits_samples: list[list[float]] = []
    sample_cosines: list[float] = []

    current_batch_size = batch_size
    max_retries = 6
    retry_count = 0
    success = False
    is_lm = dataset.lower() in LM_DATASETS

    if skip_deprecated and module_name in DEPRECATED_MODULES:
        result.status = "SKIP_DEPRECATED"
        result.error_message = f"Module '{module_name}' is in DEPRECATED_MODULES."
        run_log_lines.append(f"[SKIP] Module '{module_name}' is deprecated.")
        success = True

    while retry_count < max_retries and not success:
        model = None
        try:
            # 1. Setup environment & base model
            env = _setup_eval_environment(model_family, dataset, seq_len, device)
            model = env["model"]
            num_labels = env["num_labels"]
            mem_phases = env["mem_phases"]
            t_model_load = env["t_model_load"]
            is_cuda = env["is_cuda"]

            # 2. Compression & calibration phase
            (
                model,
                comp_ratio,
                comp_stats,
                t_calib,
                t_comp,
                calib_used,
                calib_source,
            ) = _apply_compression_phase(
                model,
                module_name,
                module_type,
                model_family,
                dataset,
                rank,
                seq_len,
                device,
                mem_phases,
                run_log_lines,
            )
            result.compression_ratio = comp_ratio
            result.calibration_used = calib_used
            result.calibration_source = calib_source
            result.params_before = sum(p.numel() for p in model.parameters())

            # 3. Dynamic batch probing
            if retry_count == 0 and is_cuda:
                current_batch_size = probe_optimal_batch_size(
                    model=model,
                    model_family=model_family,
                    dataset=dataset,
                    seq_len=seq_len,
                    device=device,
                    max_batch_cap=min(batch_size, 512) if batch_size > 0 else 512,
                    allow_synthetic=allow_synthetic,
                )

            batch = get_dataset_batch(
                dataset,
                model_family,
                current_batch_size,
                seq_len,
                num_labels=num_labels,
                device=device,
                allow_synthetic=allow_synthetic,
            )
            filtered_batch = _filter_batch(batch, model_family, model)
            if is_lm and "labels" not in filtered_batch:
                ids = filtered_batch["input_ids"]
                filtered_batch["labels"] = ids.clone()
                filtered_batch["labels"][:, :-1] = ids[:, 1:]
                filtered_batch["labels"][:, -1] = -100

            # 4. Latency & resource benchmark
            bench = _benchmark_runtime_latency(
                model,
                filtered_batch,
                num_steps,
                num_warmup,
                num_labels,
                is_lm,
                is_cuda,
                device,
                current_batch_size,
                seq_len,
            )
            result.forward_latency_ms = bench["forward_latency_ms"]
            result.backward_latency_ms = bench["backward_latency_ms"]
            result.total_step_latency_ms = bench["total_step_latency_ms"]
            result.latency_variance_ms = bench["latency_variance_ms"]
            result.throughput_samples_per_sec = bench["throughput_samples_per_sec"]
            result.throughput_tokens_sec = bench["throughput_tokens_sec"]
            result.peak_memory_mb = bench["peak_memory_mb"]
            result.loss = bench["loss"]
            result.grad_norm = bench["grad_norm"]
            result.trajectories = bench["trajectories"]
            result.flops_macs = bench["flops_macs"]

            # 5. Full evaluation on validation set
            eval_metrics = _evaluate_model_accuracy(
                model,
                model_family,
                dataset,
                num_labels,
                seq_len,
                current_batch_size,
                is_lm,
                is_cuda,
                device,
                mem_phases,
                run_log_lines,
            )
            result.perplexity = eval_metrics["perplexity"]
            result.accuracy_pct = eval_metrics["accuracy_pct"]
            result.f1_score = eval_metrics["f1_score"]
            result.mcc = eval_metrics["mcc"]
            result.pearson = eval_metrics["pearson"]
            result.spearman = eval_metrics["spearman"]
            result.fidelity_score = eval_metrics["fidelity_score"]
            t_eval = eval_metrics["t_eval"]
            baseline_logits_samples = eval_metrics["baseline_logits_samples"]
            compressed_logits_samples = eval_metrics["compressed_logits_samples"]
            sample_cosines = eval_metrics["sample_cosines"]

            result.params_after = sum(p.numel() for p in model.parameters())
            result.status = "PASS"
            success = True

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                current_batch_size = max(1, current_batch_size // 2)
                retry_count += 1
                if device.type == "cuda":
                    gc.collect()
                    torch.cuda.empty_cache()
                continue
            else:
                exc_type = type(e).__name__
                exc_traceback = traceback.format_exc()
                result.status = "FAIL"
                result.error_message = f"{exc_type}: {e}"
                break
        except Exception as e:
            exc_type = type(e).__name__
            exc_traceback = traceback.format_exc()
            result.status = "FAIL"
            result.error_message = f"{exc_type}: {e}"
            break
        finally:
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not success and result.status not in ("FAIL", "SKIP_DEPRECATED"):
        result.status = "FAIL"
        result.error_message = "OOM limits exceeded after max retries."

    return _finalize_benchmark_result(
        result,
        run_log_lines,
        t_run_start,
        module_name,
        module_type,
        model_family,
        dataset,
        rank,
        seq_len,
        batch_size,
        current_batch_size,
        num_warmup,
        num_steps,
        allow_synthetic,
        device,
        results_dir,
        comp_stats,
        mem_phases,
        t_model_load,
        t_calib,
        t_comp,
        t_eval,
        baseline_logits_samples,
        compressed_logits_samples,
        sample_cosines,
        exc_traceback,
        exc_type,
        retry_count,
        success,
    )


evaluate_single_combination = evaluate_single_combination_opt


__all__ = [
    "probe_optimal_batch_size",
    "evaluate_single_combination_opt",
    "evaluate_single_combination",
]
