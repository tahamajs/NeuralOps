"""Hardware Roofline & Arithmetic Intensity Analyzer for NeuralOps.

Empirically benchmarks compiled @triton.jit kernels vs PyTorch references,
computes analytical FLOPs, models HBM DRAM vs in-SRAM memory traffic,
and locates operating points against hardware Ridge points.
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn


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

    # Architectural specs table
    # Values: (peak_fp16_tensor_tflops, peak_hbm_bandwidth_gb_s)
    if major == 8 and minor == 9:  # Ada Lovelace (RTX 4070, 4080, 4090, L4, L40)
        if "4090" in device_name:
            return device_name, "SM 8.9", 165.0, 1008.0
        elif "4080" in device_name:
            return device_name, "SM 8.9", 112.0, 716.8
        elif "4070" in device_name:
            # RTX 4070 Desktop: ~504 GB/s, Laptop: ~256 GB/s
            bw = 504.0 if sm_count >= 46 else 256.0
            tflops = 116.0 if sm_count >= 46 else 65.0
            return device_name, "SM 8.9", tflops, bw
        else:
            tflops = max(30.0, sm_count * 1.8)
            return device_name, "SM 8.9", tflops, 300.0

    elif major == 9 and minor == 0:  # Hopper (H100, H200, H800)
        return device_name, "SM 9.0", 989.0, 3350.0

    elif major == 8 and minor == 0:  # Ampere A100
        return device_name, "SM 8.0", 312.0, 2039.0

    elif major == 8 and minor == 6:  # Ampere Consumer (RTX 3090, 3080, 3070)
        if "3090" in device_name:
            return device_name, "SM 8.6", 71.0, 936.0
        else:
            return device_name, "SM 8.6", 40.0, 448.0

    elif major == 7 and minor == 5:  # Turing (T4, RTX 2080)
        return device_name, "SM 7.5", 65.0, 320.0

    elif major == 7 and minor == 0:  # Volta (V100)
        return device_name, "SM 7.0", 125.0, 900.0

    elif major == 6 and minor == 1:  # Pascal (GTX 1060, 1070, 1080)
        return device_name, "SM 6.1", 8.8, 192.0

    # Default fallback heuristic based on SM count
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
                    setattr(mod, "FORCE_PYTORCH_FALLBACK", flag)
                except Exception:
                    pass
            if hasattr(mod, "HAS_TRITON_GDN"):
                try:
                    setattr(mod, "HAS_TRITON_GDN", not flag)
                except Exception:
                    pass


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

    # 1. Inspect module architecture for projections and SVD ranks
    proj_flops = 0.0
    svd_intermediate_bytes = 0.0
    found_projections = 0

    # Count parameters and inspect linear/factorized modules (including custom LoRALinear, Conv, etc.)
    for name, sub in module.named_modules():
        sub_cls = sub.__class__.__name__
        if isinstance(sub, nn.Linear):
            in_f = sub.in_features
            out_f = sub.out_features
            proj_flops += 2.0 * B * T * in_f * out_f
            found_projections += 1
        elif sub_cls in ("LoRALinear", "SVDLinear", "FactorizedLinear", "DynamicSVDLinear", "DobiSVDLinearLayer", "ResSVDLinear"):
            # Check for standard weight matrix or rank factorization
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
            # Add LoRA path FLOPs if present
            if hasattr(sub, "lora_A") and hasattr(sub, "lora_B") and sub.lora_A is not None and sub.lora_B is not None:
                r_lora = sub.lora_A.shape[0] if sub.lora_A.dim() == 2 else 16
                proj_flops += 2.0 * B * T * (in_f * r_lora + r_lora * out_f)
                svd_intermediate_bytes += 2.0 * element_bytes * B * T * r_lora
        elif isinstance(sub, nn.Conv1d):
            # Conv1d projection
            k = sub.kernel_size[0]
            in_c = sub.in_channels
            out_c = sub.out_channels
            proj_flops += 2.0 * B * T * in_c * out_c * k
            found_projections += 1

    # Default projection FLOPs if module doesn't expose recognized projection layers
    if found_projections == 0:
        # Standard Q, K, V, O projections: 4 * (2 * B * T * D^2) = 8 * B * T * D^2
        proj_flops = 8.0 * B * T * D * D

    # 2. Sequence Mixing Core FLOPs & Naive HBM traffic
    name_lower = module.__class__.__name__.lower()
    if hasattr(module, "target_attention"):
        name_lower += " " + str(getattr(module, "target_attention")).lower()

    is_quadratic = any(k in name_lower for k in ("softmax", "full", "mha", "exact", "standard"))
    is_sliding = "window" in name_lower or "sliding" in name_lower
    is_linear_recurrent = any(
        k in name_lower for k in ("linear", "gdn", "delta", "gla", "mamba", "ssm", "hgrn", "decay", "kda", "cosformer")
    )

    if is_quadratic and not is_sliding:
        # Softmax Q @ K.T (2 * B * H * T^2 * d) + Attn @ V (2 * B * H * T^2 * d)
        core_flops = 4.0 * B * H * T * T * d
        naive_attention_traffic = 4.0 * element_bytes * B * H * T * T
    elif is_sliding:
        win = 256
        core_flops = 4.0 * B * H * T * min(T, win) * d
        naive_attention_traffic = 4.0 * element_bytes * B * H * T * min(T, win)
    elif is_linear_recurrent:
        # State update S_t = S_{t-1} + k_t * v_t^T (2 * B * H * T * d^2) + q_t * S_t (2 * B * H * T * d^2)
        core_flops = 4.0 * B * H * T * d * d
        # Naive PyTorch allocates recurrent state across all time: B x H x T x d x d
        naive_attention_traffic = 2.0 * element_bytes * B * H * T * d * d
    else:
        # Default linear representation
        core_flops = 4.0 * B * H * T * d * d
        naive_attention_traffic = 2.0 * element_bytes * B * H * T * d * d

    total_flops = float(proj_flops + core_flops)

    # 3. Model parameter bytes
    param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    if param_bytes == 0:
        param_bytes = 4 * D * D * element_bytes

    # 4. HBM Traffic Calculations
    # Fused Triton Path: reads input X, reads weights, writes output Y (+ minimal boundaries)
    io_bytes = 2.0 * element_bytes * B * T * D  # Input X and Output Y
    qkv_boundaries = 3.0 * element_bytes * B * T * D  # Q, K, V activations between projection & attention
    hbm_bytes_fused = float(io_bytes + param_bytes + qkv_boundaries)

    # Naive PyTorch Path: additionally materializes all intermediate factorizations, attention matrices, and recurrent states
    hbm_bytes_naive = float(hbm_bytes_fused + naive_attention_traffic + svd_intermediate_bytes)

    return total_flops, hbm_bytes_naive, hbm_bytes_fused


def profile_roofline(
    module_or_name: Union[str, nn.Module],
    batch_size: int = 2,
    seq_len: int = 2048,
    embed_dim: int = 768,
    num_heads: int = 12,
    dtype: str = "float16",
    warmup: int = 10,
    rep: int = 20,
    device: Optional[Union[str, torch.device]] = None,
    **module_kwargs: Any,
) -> RooflineReport:
    """Run dual-path timing (Triton vs PyTorch) and compute Roofline arithmetic intensity."""
    # 1. Resolve Device
    if device is None:
        target_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    # 2. Resolve Dtype
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
        # Fallback to float32 on CPU for stability if float16 not natively supported
        torch_dtype = torch.float32
        element_bytes = 4

    # 3. Instantiate or resolve module
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
                except Exception:
                    try:
                        from transformers import BertConfig
                        cfg = BertConfig(hidden_size=embed_dim, num_attention_heads=num_heads)
                        module = cls(config=cfg, **module_kwargs)
                    except Exception:
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

    # Helper forward invocation
    def _run_forward(m: nn.Module, inp: torch.Tensor) -> torch.Tensor:
        try:
            out = m(inp)
        except TypeError:
            try:
                out = m(hidden_states=inp)
            except TypeError:
                out = m(inp, inp, inp)
        return out[0] if isinstance(out, tuple) else out

    # 4. Measure Peak VRAM & Dual-Path Timing
    is_cuda = target_device.type == "cuda"
    latency_triton_ms = 0.0
    latency_pytorch_ms = 0.0
    peak_vram_mb = 0.0

    try:
        # A. Warmup & Measure Triton Path (FORCE_PYTORCH_FALLBACK = False)
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

        # B. Measure PyTorch Reference Path (FORCE_PYTORCH_FALLBACK = True)
        _set_global_fallback(True)
        py_warmup = min(3, warmup)
        py_rep = min(5, rep)
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
        _set_global_fallback(False)

    # 5. Roofline Calculations
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

    # Bottleneck classification
    if ai_fused >= 0.85 * ridge_point:
        regime = "Compute-Bound (Tensor Core Saturated)"
    elif ai_fused > 1.5 * ai_naive:
        regime = "SRAM-Fused Transition Regime (Memory Wall Mitigated)"
    else:
        regime = "Memory-Bandwidth Bound (HBM Limited)"

    return RooflineReport(
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


def render_roofline_terminal(report: RooflineReport) -> str:
    """Render an ASCII Roofline model chart and diagnostic summary."""
    lines = []
    w = 78
    lines.append("=" * w)
    lines.append(f"  🏛️ NEURALOPS HARDWARE ROOFLINE & ARITHMETIC INTENSITY REPORT")
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
    lines.append("-" * w)
    lines.append(f"  Latency (PyTorch Native): {report.latency_pytorch_ms:>8.3f} ms")
    lines.append(f"  Latency (NeuralOps Triton):{report.latency_triton_ms:>8.3f} ms  ==>  🚀 SPEEDUP: {report.speedup_x:>6.2f}x")
    lines.append(f"  Achieved Performance:     {report.achieved_tflops:>8.2f} TFLOPS ({report.achieved_bandwidth_gb_s:.1f} GB/s HBM bandwidth)")
    lines.append("=" * w)

    # -------------------------------------------------------------------------
    # Dynamic 2D Log-Log ASCII Roofline Chart (14 rows x 60 cols)
    # -------------------------------------------------------------------------
    plot_rows = 14
    plot_cols = 58
    # Log10 ranges: AI from 0.25 to 1024 -> [-0.60, 3.01]
    min_log_ai = math.log10(0.25)
    max_log_ai = math.log10(1024.0)

    # Performance Y-axis: Log10 TFLOPS from 0.05 to peak_tflops * 1.2
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

    # Draw Roofline Curve:
    # For cols < ridge_col: attainable = min(peak_tflops, ai * peak_bw / 1000)
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

    # Plot [P] (Naive PyTorch Point)
    py_tflops = (report.total_flops / (max(1e-5, report.latency_pytorch_ms / 1000.0) * 1e12))
    p_row = _perf_to_row(py_tflops)
    p_col = _ai_to_col(report.arithmetic_intensity_naive)
    if 0 <= p_row < plot_rows and 0 <= p_col < plot_cols:
        grid[p_row][p_col] = "P"

    # Plot [T] (Fused Triton Point)
    t_row = _perf_to_row(report.achieved_tflops)
    t_col = _ai_to_col(report.arithmetic_intensity_fused)
    if 0 <= t_row < plot_rows and 0 <= t_col < plot_cols:
        grid[t_row][t_col] = "T"

    lines.append("  LOG-LOG ROOFLINE HARDWARE CHARACTERISTIC CURVE:")
    lines.append(f"  TFLOPS (Peak: {report.peak_tflops:.1f} TFLOPS │ Ridge: {report.ridge_point:.1f} FLOPs/B │ Slope: {report.peak_bandwidth_gb_s:.1f} GB/s)")
    lines.append("    ^")
    for r_idx, row_chars in enumerate(grid):
        # Y-axis tick labels
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
