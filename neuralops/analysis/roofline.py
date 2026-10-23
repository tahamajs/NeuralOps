#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Roofline Model Analyzer

Empirical arithmetic intensity and memory bandwidth analyzer benchmarking theoretical vs attained FLOP/s.
"""

from __future__ import annotations

import gc
import logging
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class RooflineReport:
    """Complete diagnostic report from hardware Roofline analysis."""

    module_name: str
    device_name: str
    sm_version: str
    precision: str
    batch_size: int
    seq_len: int
    embed_dim: int
    num_heads: int
    head_dim: int
    total_flops: float
    hbm_bytes_naive: float
    hbm_bytes_fused: float
    sram_bytes_saved_mb: float
    arithmetic_intensity_naive: float
    arithmetic_intensity_fused: float
    peak_tflops: float
    peak_bandwidth_gb_s: float
    ridge_point: float
    latency_pytorch_ms: float
    latency_triton_ms: float
    speedup_x: float
    achieved_tflops: float
    achieved_bandwidth_gb_s: float
    peak_vram_mb: float
    bottleneck_regime: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary representation."""
        return asdict(self)


def get_device_roofline_specs(device: Optional[Union[str, torch.device]] = None) -> Tuple[str, str, float, float]:
    """Retrieve peak TFLOPS (FP16/BF16) and peak HBM Bandwidth (GB/s) for device.

    Returns:
        (device_name, sm_version, peak_tflops, peak_bandwidth_gb_s)
    """
    if not torch.cuda.is_available():
        return "CPU Host", "cpu", 2.0, 50.0

    target_device = torch.device(device if device is not None else "cuda:0")
    if target_device.type != "cuda":
        return "CPU Host", "cpu", 2.0, 50.0

    props = torch.cuda.get_device_properties(target_device)
    major, minor = props.major, props.minor
    sm_version = f"{major}.{minor}"
    device_name = props.name
    sm_count = getattr(props, "multi_processor_count", 36)

    if major == 8 and minor == 9:
        if "4090" in device_name:
            return device_name, "SM 8.9", 165.0, 1008.0
        elif "4080" in device_name:
            return device_name, "SM 8.9", 112.0, 716.8
        elif "4070" in device_name:
            bw = 504.0 if sm_count >= 46 else 256.0
            tflops = 116.0 if sm_count >= 46 else 65.0
            return device_name, "SM 8.9", tflops, bw
        else:
            tflops = max(30.0, sm_count * 1.8)
            return device_name, "SM 8.9", tflops, 300.0

    elif major == 9 and minor == 0:
        return device_name, "SM 9.0", 989.0, 3350.0

    elif major == 8 and minor == 0:
        return device_name, "SM 8.0", 312.0, 2039.0

    elif major == 8 and minor == 6:
        if "3090" in device_name:
            return device_name, "SM 8.6", 71.0, 936.0
        else:
            return device_name, "SM 8.6", 40.0, 448.0

    elif major == 7 and minor == 5:
        return device_name, "SM 7.5", 65.0, 320.0

    elif major == 7 and minor == 0:
        return device_name, "SM 7.0", 125.0, 900.0

    elif major == 6 and minor == 1:
        return device_name, "SM 6.1", 8.8, 192.0

    est_tflops = float(max(10.0, sm_count * 1.5))
    est_bw = float(max(100.0, sm_count * 6.0))
    return device_name, f"SM {sm_version}", est_tflops, est_bw


def _set_global_fallback(flag: bool) -> None:
    """Recursively set FORCE_PYTORCH_FALLBACK across all loaded neuralops modules."""
    import sys
    from neuralops.core.triton_kernels import _types as tt_types
    tt_types.FORCE_PYTORCH_FALLBACK = flag

    for name, mod in list(sys.modules.items()):
        if name.startswith("neuralops.") and mod is not None:
            if hasattr(mod, "FORCE_PYTORCH_FALLBACK"):
                try:
                    mod.FORCE_PYTORCH_FALLBACK = flag
                except Exception as exc:
                    logger.debug("Could not set FORCE_PYTORCH_FALLBACK on module %s: %s", name, exc)
            if hasattr(mod, "HAS_TRITON_GDN"):
                try:
                    mod.HAS_TRITON_GDN = not flag
                except Exception as exc:
                    logger.debug("Could not set HAS_TRITON_GDN on module %s: %s", name, exc)


def estimate_module_flops_and_bytes(
    module: nn.Module,
    batch_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    element_bytes: int = 2,
) -> Tuple[float, float, float]:
    """Analytical FLOPs and HBM traffic estimation (Naive vs Fused).

    Returns:
        (total_flops, hbm_bytes_naive, hbm_bytes_fused)
    """
    head_dim = max(1, embed_dim // max(1, num_heads))
    B, T, D, H, d = batch_size, seq_len, embed_dim, num_heads, head_dim

    proj_flops = 0.0
    svd_intermediate_bytes = 0.0
    found_projections = 0

    for name, sub in module.named_modules():
        sub_cls = sub.__class__.__name__
        if isinstance(sub, nn.Linear):
            in_f = sub.in_features
            out_f = sub.out_features
            proj_flops += 2.0 * B * T * in_f * out_f
            found_projections += 1
        elif sub_cls in ("LoRALinear", "SVDLinear", "FactorizedLinear", "DynamicSVDLinear", "DobiSVDLinearLayer", "ResSVDLinear"):
            if hasattr(sub, "weight") and sub.weight is not None and hasattr(sub.weight, "shape") and len(sub.weight.shape) == 2:
                out_f, in_f = sub.weight.shape
                proj_flops += 2.0 * B * T * in_f * out_f
                found_projections += 1
            else:
                in_f = getattr(sub, "in_features", D)
                out_f = getattr(sub, "out_features", D)
                r = getattr(sub, "rank", getattr(sub, "current_rank", getattr(sub, "r", 32)))
                proj_flops += 2.0 * B * T * (in_f * r + r * out_f)
                svd_intermediate_bytes += 2.0 * element_bytes * B * T * r
                found_projections += 1
            if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
                r_lora = sub.lora_A.shape[0] if sub.lora_A.dim() == 2 else 16
                proj_flops += 2.0 * B * T * (in_f * r_lora + r_lora * out_f)
                svd_intermediate_bytes += 2.0 * element_bytes * B * T * r_lora
        elif isinstance(sub, nn.Conv1d):
            k = sub.kernel_size[0]
            in_c = sub.in_channels
            out_c = sub.out_channels
            proj_flops += 2.0 * B * T * in_c * out_c * k
            found_projections += 1

    if found_projections == 0:
        proj_flops = 8.0 * B * T * D * D

    name_lower = module.__class__.__name__.lower()
    if hasattr(module, "target_attention"):
        name_lower += " " + str(module.target_attention).lower()

    is_quadratic = any(k in name_lower for k in ("softmax", "full", "mha", "exact", "standard"))
    is_sliding = "window" in name_lower or "sliding" in name_lower
    is_linear_recurrent = any(
        k in name_lower for k in ("linear", "gdn", "delta", "gla", "mamba", "ssm", "hgrn", "decay", "kda", "cosformer")
    )

    if is_quadratic and not is_sliding:
        core_flops = 4.0 * B * H * T * T * d
        naive_attention_traffic = 4.0 * element_bytes * B * H * T * T
    elif is_sliding:
        win = 256
        core_flops = 4.0 * B * H * T * min(T, win) * d
        naive_attention_traffic = 4.0 * element_bytes * B * H * T * min(T, win)
    elif is_linear_recurrent:
        core_flops = 4.0 * B * H * T * d * d
        naive_attention_traffic = 2.0 * element_bytes * B * H * T * d * d
    else:
        core_flops = 4.0 * B * H * T * d * d
        naive_attention_traffic = 2.0 * element_bytes * B * H * T * d * d

    total_flops = float(proj_flops + core_flops)

    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    if param_bytes == 0:
        param_bytes = 4 * D * D * element_bytes

    io_bytes = 2.0 * element_bytes * B * T * D
    qkv_boundaries = 3.0 * element_bytes * B * T * D
    hbm_bytes_fused = float(io_bytes + param_bytes + qkv_boundaries)

    hbm_bytes_naive = float(hbm_bytes_fused + naive_attention_traffic + svd_intermediate_bytes)

    return total_flops, hbm_bytes_naive, hbm_bytes_fused


def profile_roofline(
    module_or_name: Optional[Union[str, nn.Module]] = None,
    batch_size: int = 2,
    seq_len: int = 2048,
    embed_dim: int = 768,
    num_heads: int = 12,
    dtype: str = "float16",
    warmup: int = 10,
    rep: int = 10,
    device: Optional[Union[str, torch.device]] = None,
    module_name: Optional[str] = None,
    log_to_ledger: bool = True,
    **module_kwargs: Any,
) -> RooflineReport:
    """Run dual-path timing (Triton vs PyTorch) and compute Roofline arithmetic intensity."""
    if module_or_name is None:
        if module_name is not None:
            module_or_name = module_name
        else:
            module_or_name = "gated_delta_net"
    if device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    dtype_map = {
        "float16": (torch.float16, 2),
        "fp16": (torch.float16, 2),
        "bfloat16": (torch.bfloat16, 2),
        "bf16": (torch.bfloat16, 2),
        "float32": (torch.float32, 4),
        "fp32": (torch.float32, 4),
    }
    torch_dtype, element_bytes = dtype_map.get(dtype.lower(), (torch.float16, 2))
    if target_device.type == "cpu" and torch_dtype == torch.float16:
        torch_dtype = torch.float32
        element_bytes = 4

    if isinstance(module_or_name, str):
        module_name = module_or_name
        if "compose(" in module_name or "proj=" in module_name:
            from neuralops.core.composable import ComposableAttention
            module = ComposableAttention(embed_dim=embed_dim, num_heads=num_heads, **module_kwargs)
        else:
            from neuralops.core.registry import ATTENTION_REGISTRY
            cls = ATTENTION_REGISTRY.get(module_name)
            if cls is not None:
                try:
                    module = cls(embed_dim=embed_dim, num_heads=num_heads, **module_kwargs)
                except Exception as exc1:
                    logger.debug("Failed initializing %s with embed_dim/num_heads: %s; trying BertConfig", module_name, exc1)
                    try:
                        from transformers import BertConfig
                        cfg = BertConfig(hidden_size=embed_dim, num_attention_heads=num_heads)
                        module = cls(config=cfg, **module_kwargs)
                    except Exception as exc2:
                        logger.debug("Failed initializing %s with BertConfig: %s; trying kwargs only", module_name, exc2)
                        module = cls(**module_kwargs)
            else:
                from neuralops.core.composable import ComposableAttention
                module = ComposableAttention(embed_dim=embed_dim, num_heads=num_heads)
    else:
        module = module_or_name
        module_name = module.__class__.__name__

    module = module.to(device=target_device, dtype=torch_dtype)
    module.eval()

    head_dim = max(1, embed_dim // max(1, num_heads))
    x = torch.randn(batch_size, seq_len, embed_dim, device=target_device, dtype=torch_dtype)

    def _run_forward(m: nn.Module, inp: torch.Tensor) -> torch.Tensor:
        try:
            out = m(inp)
        except TypeError as exc1:
            logger.debug("Forward failed with single input: %s; trying hidden_states", exc1)
            try:
                out = m(hidden_states=inp)
            except TypeError as exc2:
                logger.debug("Forward failed with hidden_states: %s; trying 3-arg forward", exc2)
                out = m(inp, inp, inp)
        return out[0] if isinstance(out, tuple) else out

    is_cuda = target_device.type == "cuda"
    latency_triton_ms = 0.0
    latency_pytorch_ms = 0.0
    peak_vram_mb = 0.0

    try:
        _set_global_fallback(False)
        with torch.no_grad():
            for _ in range(warmup):
                _ = _run_forward(module, x)
        if is_cuda:
            torch.cuda.synchronize(target_device)

        if is_cuda:
            torch.cuda.reset_peak_memory_stats(target_device)
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            with torch.no_grad():
                for _ in range(rep):
                    _ = _run_forward(module, x)
            end_evt.record()
            torch.cuda.synchronize(target_device)
            latency_triton_ms = float(start_evt.elapsed_time(end_evt) / rep)
            peak_vram_mb = float(torch.cuda.max_memory_allocated(target_device) / (1024.0 * 1024.0))
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                for _ in range(rep):
                    _ = _run_forward(module, x)
            latency_triton_ms = float((time.perf_counter() - t0) * 1000.0 / rep)
            peak_vram_mb = 0.0

        _set_global_fallback(True)
        _saved_fla_chunk = None
        try:
            import sys
            _fla_ops = sys.modules.get("fla.ops")
            if _fla_ops is not None and hasattr(_fla_ops, "chunk_gdn"):
                _saved_fla_chunk = _fla_ops.chunk_gdn
                delattr(_fla_ops, "chunk_gdn")
        except Exception as exc:
            logger.debug("Could not detach fla.ops.chunk_gdn: %s", exc)

        py_warmup = min(2, warmup)
        py_rep = min(3, rep)
        try:
            with torch.no_grad():
                for _ in range(py_warmup):
                    _ = _run_forward(module, x)
            if is_cuda:
                torch.cuda.synchronize(target_device)

            if is_cuda:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                with torch.no_grad():
                    for _ in range(py_rep):
                        _ = _run_forward(module, x)
                end_evt.record()
                torch.cuda.synchronize(target_device)
                latency_pytorch_ms = float(start_evt.elapsed_time(end_evt) / py_rep)
            else:
                t0 = time.perf_counter()
                with torch.no_grad():
                    for _ in range(py_rep):
                        _ = _run_forward(module, x)
                latency_pytorch_ms = float((time.perf_counter() - t0) * 1000.0 / py_rep)
        finally:
            if _saved_fla_chunk is not None:
                try:
                    import sys
                    _fla_ops = sys.modules.get("fla.ops")
                    if _fla_ops is not None:
                        _fla_ops.chunk_gdn = _saved_fla_chunk
                except Exception as exc:
                    logger.debug("Could not restore fla.ops.chunk_gdn: %s", exc)

    finally:
        _set_global_fallback(False)

    total_flops, hbm_bytes_naive, hbm_bytes_fused = estimate_module_flops_and_bytes(
        module=module,
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        element_bytes=element_bytes,
    )

    sram_bytes_saved_mb = max(0.0, (hbm_bytes_naive - hbm_bytes_fused) / (1024.0 * 1024.0))
    ai_naive = total_flops / max(1.0, hbm_bytes_naive)
    ai_fused = total_flops / max(1.0, hbm_bytes_fused)

    device_name, sm_version, peak_tflops, peak_bw = get_device_roofline_specs(target_device)
    ridge_point = (peak_tflops * 1000.0) / max(1.0, peak_bw)

    speedup_x = latency_pytorch_ms / max(1e-5, latency_triton_ms)
    t_sec = latency_triton_ms / 1000.0
    achieved_tflops = (total_flops / (t_sec * 1e12)) if t_sec > 0 else 0.0
    achieved_bandwidth_gb_s = (hbm_bytes_fused / (t_sec * 1e9)) if t_sec > 0 else 0.0

    if ai_fused >= 0.85 * ridge_point:
        regime = "Compute-Bound (Tensor Core Saturated)"
    elif ai_fused > 1.5 * ai_naive:
        regime = "SRAM-Fused Transition Regime (Memory Wall Mitigated)"
    else:
        regime = "Memory-Bandwidth Bound (HBM Limited)"

    rep_obj = RooflineReport(
        module_name=module_name,
        device_name=device_name,
        sm_version=sm_version,
        precision=dtype.upper(),
        batch_size=batch_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_heads=num_heads,
        head_dim=head_dim,
        total_flops=total_flops,
        hbm_bytes_naive=hbm_bytes_naive,
        hbm_bytes_fused=hbm_bytes_fused,
        sram_bytes_saved_mb=sram_bytes_saved_mb,
        arithmetic_intensity_naive=ai_naive,
        arithmetic_intensity_fused=ai_fused,
        peak_tflops=peak_tflops,
        peak_bandwidth_gb_s=peak_bw,
        ridge_point=ridge_point,
        latency_pytorch_ms=latency_pytorch_ms,
        latency_triton_ms=latency_triton_ms,
        speedup_x=speedup_x,
        achieved_tflops=achieved_tflops,
        achieved_bandwidth_gb_s=achieved_bandwidth_gb_s,
        peak_vram_mb=peak_vram_mb,
        bottleneck_regime=regime,
    )

    if log_to_ledger:
        try:
            from neuralops.analysis.results_ledger import log_experiment_result
            log_experiment_result(
                module_name=module_name,
                category="triton_kernel" if any(k in module_name.lower() for k in ("triton", "kernel", "scan")) else "attention",
                family="hardware-roofline",
                generation_method={
                    "algorithm": "Hardware Roofline Dual-Path CUDA Event Profiler",
                    "source_component": "neuralops.analysis.roofline.profile_roofline",
                },
                experimental_setup={
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "dtype": dtype,
                    "hardware": f"{device_name} ({sm_version})",
                },
                metrics={
                    "latency_triton_ms": latency_triton_ms,
                    "latency_pytorch_ms": latency_pytorch_ms,
                    "speedup_x": speedup_x,
                    "achieved_tflops": achieved_tflops,
                    "achieved_bandwidth_gb_s": achieved_bandwidth_gb_s,
                    "peak_vram_mb": peak_vram_mb,
                    "sram_bytes_saved_mb": sram_bytes_saved_mb,
                    "arithmetic_intensity_fused": ai_fused,
                },
                notes=f"Bottleneck regime: {regime}",
            )
        except Exception as exc:
            logger.warning("Failed to record experiment result to ledger for %s: %s", module_name, exc)

    return rep_obj


def render_roofline_terminal(report: RooflineReport) -> str:
    """Render an ASCII Roofline model chart and diagnostic summary."""
    lines = []
    w = 78
    lines.append("=" * w)
    lines.append("  🏛️ NEURALOPS HARDWARE ROOFLINE & ARITHMETIC INTENSITY REPORT")
    lines.append("=" * w)
    lines.append(f"  Module:          {report.module_name:<28} Precision:   {report.precision}")
    lines.append(f"  Device:          {report.device_name} ({report.sm_version})")
    lines.append(f"  Input Shape:     B={report.batch_size}, T={report.seq_len}, D={report.embed_dim}, H={report.num_heads} (head_dim={report.head_dim})")
    lines.append("-" * w)
    lines.append(f"  Peak Compute:    {report.peak_tflops:>8.1f} TFLOPS          Peak Bandwidth: {report.peak_bandwidth_gb_s:>8.1f} GB/s")
    lines.append(f"  Hardware Ridge:  {report.ridge_point:>8.2f} FLOPs/Byte      Bottleneck:     {report.bottleneck_regime}")
    lines.append("-" * w)
    lines.append(f"  Total Work:      {report.total_flops / 1e9:>8.2f} GFLOPs")
    lines.append(f"  HBM Naive:       {report.hbm_bytes_naive / 1e6:>8.2f} MB (AI: {report.arithmetic_intensity_naive:.2f} FLOPs/B)")
    lines.append(f"  HBM Fused:       {report.hbm_bytes_fused / 1e6:>8.2f} MB (AI: {report.arithmetic_intensity_fused:.2f} FLOPs/B)")
    lines.append(f"  SRAM Traffic Saved by Triton Fusion: {report.sram_bytes_saved_mb:.2f} MB")
    py_tflops = (report.total_flops / (max(1e-5, report.latency_pytorch_ms / 1000.0) * 1e12))
    py_bw = (report.hbm_bytes_naive / (max(1e-5, report.latency_pytorch_ms / 1000.0) * 1e9))
    lines.append(f"  Latency (PyTorch Native): {report.latency_pytorch_ms:>8.3f} ms  -->  Achieved [P]: {py_tflops:>6.2f} TFLOPS ({py_bw:>6.1f} GB/s)")
    lines.append(f"  Latency (NeuralOps Triton):{report.latency_triton_ms:>8.3f} ms  -->  Achieved [T]: {report.achieved_tflops:>6.2f} TFLOPS ({report.achieved_bandwidth_gb_s:>6.1f} GB/s)")
    lines.append(f"  🚀 EMPIRICAL SPEEDUP:      {report.speedup_x:>8.2f}x (Hardware Efficiency Gain)")
    lines.append("=" * w)

    plot_rows = 14
    plot_cols = 58
    min_log_ai = math.log10(0.25)
    max_log_ai = math.log10(1024.0)

    min_tflops = 0.05
    max_tflops = max(10.0, report.peak_tflops * 1.25)
    min_log_p = math.log10(min_tflops)
    max_log_p = math.log10(max_tflops)

    grid = [[" " for _ in range(plot_cols)] for _ in range(plot_rows)]

    def _ai_to_col(ai_val: float) -> int:
        l_ai = math.log10(max(0.25, min(1024.0, ai_val)))
        frac = (l_ai - min_log_ai) / (max_log_ai - min_log_ai)
        c = int(round(frac * (plot_cols - 1)))
        return max(0, min(plot_cols - 1, c))

    def _perf_to_row(p_val: float) -> int:
        l_p = math.log10(max(min_tflops, min(max_tflops, p_val)))
        frac = (l_p - min_log_p) / (max_log_p - min_log_p)
        r = int(round((1.0 - frac) * (plot_rows - 1)))
        return max(0, min(plot_rows - 1, r))

    ridge_col = _ai_to_col(report.ridge_point)
    peak_row = _perf_to_row(report.peak_tflops)

    for c in range(plot_cols):
        l_ai = min_log_ai + (c / (plot_cols - 1)) * (max_log_ai - min_log_ai)
        ai_at_col = 10.0 ** l_ai
        attainable_p = min(report.peak_tflops, (ai_at_col * report.peak_bandwidth_gb_s) / 1000.0)
        r = _perf_to_row(attainable_p)
        if c >= ridge_col:
            grid[peak_row][c] = "─"
        else:
            grid[r][c] = "/"

    if 0 <= peak_row < plot_rows and 0 <= ridge_col < plot_cols:
        grid[peak_row][ridge_col] = "┬"

    p_row = _perf_to_row(py_tflops)
    p_col = _ai_to_col(report.arithmetic_intensity_naive)
    if 0 <= p_row < plot_rows and 0 <= p_col < plot_cols:
        grid[p_row][p_col] = "P"

    t_row = _perf_to_row(report.achieved_tflops)
    t_col = _ai_to_col(report.arithmetic_intensity_fused)
    if 0 <= t_row < plot_rows and 0 <= t_col < plot_cols:
        grid[t_row][t_col] = "T"

    lines.append("  LOG-LOG ROOFLINE HARDWARE CHARACTERISTIC CURVE:")
    lines.append(f"  TFLOPS (Peak: {report.peak_tflops:.1f} TFLOPS │ Ridge: {report.ridge_point:.1f} FLOPs/B │ Slope: {report.peak_bandwidth_gb_s:.1f} GB/s)")
    lines.append("    ^")
    for r_idx, row_chars in enumerate(grid):
        if r_idx == peak_row:
            lbl = f"{report.peak_tflops:>5.1f} |"
        elif r_idx == 0:
            lbl = f"{max_tflops:>5.1f} |"
        elif r_idx == plot_rows - 1:
            lbl = f"{min_tflops:>5.2f} |"
        else:
            lbl = "      |"
        lines.append(f"  {lbl} " + "".join(row_chars))

    lines.append("      +-" + "─" * plot_cols + "-> Arithmetic Intensity")
    lines.append("        0.25         1.0         4.0        16.0        64.0       256.0      1024.0 (FLOPs/Byte)")
    lines.append("     Legend: [P] Naive PyTorch Path       [T] NeuralOps Fused Triton Kernel")
    lines.append("=" * w)
    return "\n".join(lines)


__all__ = [
    "RooflineReport",
    "get_device_roofline_specs",
    "estimate_module_flops_and_bytes",
    "profile_roofline",
    "render_roofline_terminal",
]
