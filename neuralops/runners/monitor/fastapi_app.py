"""FastAPI Application providing modern REST & streaming endpoints for NeuralOps Dashboard."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from neuralops.runners.monitor.schemas import (
    ApiResponse,
    AttentionModuleItem,
    CompressorBenchmarkItem,
    ExperimentLaunchRequest,
    GpuMetrics,
    PagedKVCacheAllocation,
    SpeculativeDecodingMetrics,
)
from neuralops.runners.monitor.advanced_data import (
    get_all_131_attentions,
    get_all_72_compressors,
    get_paged_kv_cache_state,
    get_speculative_decoding_state,
)
from neuralops.runners.monitor.stats import build_aggregated_stats
from neuralops.runners.monitor.telemetry import get_live_gpu_telemetry, get_live_logs
from neuralops.runners.monitor.endpoints import (
    get_doc_content,
    get_loss_history,
    get_module_detail,
    launch_new_run,
    stop_current_run,
)


def create_app(project_dir: Optional[str] = None) -> FastAPI:
    """Create and configure the FastAPI web monitor application."""
    if project_dir is None:
        project_dir = os.environ.get("PROJECT_DIR", os.getcwd())

    docs_dir = os.path.join(project_dir, "docs")
    harvested_json = os.path.join(project_dir, "results", "harvested_benchmark_data.json")

    app = FastAPI(
        title="NeuralOps Research Platform & Telemetry Monitor",
        description="Production REST API and real-time streaming dashboard for subquadratic attention mechanisms, compression leaderboards, PagedKVCache allocation, and speculative decoding.",
        version="3.5.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=ApiResponse, tags=["System"])
    async def health_check() -> ApiResponse:
        """Health check endpoint validating server uptime."""
        return ApiResponse(success=True, message="NeuralOps Telemetry Server Healthy")

    @app.get("/api/gpu/metrics", response_model=GpuMetrics, tags=["Telemetry"])
    async def get_gpu_metrics() -> GpuMetrics:
        """Fetch live GPU memory, utilization, temperature, and power metrics."""
        data = get_live_gpu_telemetry(project_dir)
        mem_used = float(data.get("mem_used_mb", 0.0))
        mem_total = float(data.get("mem_total_mb", 8192.0))
        mem_pct = round((mem_used / max(1.0, mem_total)) * 100.0, 1)
        return GpuMetrics(
            name=data.get("gpu_name", "NVIDIA GeForce RTX 4070 (8GB)"),
            gpu_util=float(data.get("gpu_util", 0.0)),
            mem_used_mb=mem_used,
            mem_total_mb=mem_total,
            mem_pct=mem_pct,
            power_w=float(data.get("power_w", 0.0)),
            temp_c=float(data.get("temp_c", 0.0)),
        )

    @app.get("/api/events/live", tags=["Streaming"])
    async def sse_live_telemetry() -> StreamingResponse:
        """Server-Sent Events (SSE) streaming real-time GPU telemetry, batching stats, and speculative acceptance."""
        async def event_generator() -> AsyncGenerator[str, None]:
            while True:
                gpu_data = get_live_gpu_telemetry(project_dir)
                spec = get_speculative_decoding_state()
                paged = get_paged_kv_cache_state()
                payload = {
                    "gpu": gpu_data,
                    "speculative": {
                        "acceptance_rate": spec["acceptance_rate"],
                        "mean_accepted_tokens": spec["mean_accepted_tokens"],
                        "speedup_factor": spec["speedup_factor"],
                    },
                    "paged_kv": {
                        "utilization_pct": paged["utilization_pct"],
                        "allocated_blocks": paged["allocated_blocks"],
                        "free_blocks": paged["free_blocks"],
                    },
                }
                yield f"data: {json.dumps(payload)}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/api/attentions", response_model=List[AttentionModuleItem], tags=["Attentions"])
    async def list_131_attentions() -> List[AttentionModuleItem]:
        """Retrieve complete directory of all 131 attention mechanisms with latencies and metrics."""
        raw_items = get_all_131_attentions(project_dir)
        return [AttentionModuleItem(**item) for item in raw_items]

    @app.get("/api/compressors", response_model=List[CompressorBenchmarkItem], tags=["Compression"])
    async def list_72_compressors() -> List[CompressorBenchmarkItem]:
        """Retrieve Pareto leaderboard of all 72 compression engines sorted by reconstruction accuracy."""
        raw_items = get_all_72_compressors(project_dir)
        return [CompressorBenchmarkItem(**item) for item in raw_items]

    @app.get("/api/paged_kv_cache", response_model=PagedKVCacheAllocation, tags=["Inference"])
    async def inspect_paged_kv_cache() -> PagedKVCacheAllocation:
        """Inspect continuous batching and PagedKVCache block allocation table."""
        return PagedKVCacheAllocation(**get_paged_kv_cache_state())

    @app.get("/api/speculative_decoding", response_model=SpeculativeDecodingMetrics, tags=["Inference"])
    async def inspect_speculative_decoding() -> SpeculativeDecodingMetrics:
        """Track token acceptance rate and acceleration curves for self-speculative decoding."""
        return SpeculativeDecodingMetrics(**get_speculative_decoding_state())

    @app.get("/api/stats", tags=["Leaderboard"])
    async def get_overview_stats() -> Dict[str, Any]:
        """Aggregate cross-model validation stats, pass/fail rates, and leaderboards."""
        return build_aggregated_stats(project_dir, harvested_json, docs_dir)

    @app.get("/api/loss_history", tags=["Training"])
    async def loss_trajectory(module: Optional[str] = None, model: Optional[str] = None) -> Dict[str, Any]:
        """Return step-by-step loss history and convergence trajectory."""
        return get_loss_history(project_dir, module=module, model=model)

    @app.get("/api/module_detail", tags=["Modules"])
    async def module_detail(module: str = Query("hgrn2_gated_deltanet")) -> Dict[str, Any]:
        """Detailed documentation, dataset accuracy breakdown, and profile for an attention module."""
        return get_module_detail(docs_dir, project_dir, module)

    @app.get("/api/doc", tags=["Documentation"])
    async def doc_content(module: str = Query("svd_linear_attention")) -> Dict[str, Any]:
        """Fetch raw markdown documentation content for a given module."""
        return get_doc_content(docs_dir, project_dir, module)

    @app.get("/api/logs", tags=["Telemetry"])
    async def fetch_logs(source: str = "active", lines: int = 100) -> Dict[str, Any]:
        """Fetch recent console or execution log lines."""
        return get_live_logs(project_dir, source=source, lines=lines)

    @app.post("/api/run/launch", response_model=ApiResponse, tags=["Execution"])
    async def launch_sweep(req: ExperimentLaunchRequest) -> ApiResponse:
        """Launch or queue a new benchmark or distillation run."""
        res = launch_new_run(req.dict())
        return ApiResponse(success=res.get("success", True), message="Benchmark run queued", data=res)

    @app.post("/api/run/stop", response_model=ApiResponse, tags=["Execution"])
    async def stop_sweep() -> ApiResponse:
        """Safely halt the currently active training or benchmark process."""
        res = stop_current_run()
        return ApiResponse(success=res.get("success", True), message="Run termination signal sent", data=res)

    # Static assets mounting
    if os.path.exists(docs_dir):
        app.mount("/docs_static", StaticFiles(directory=docs_dir, html=False), name="docs_static")

    return app
