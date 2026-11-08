# NeuralOps Codebase Architecture: Part 09
# Hardware Telemetry, Microbenchmarking & Roofline Engine

**Assigned Subagent**: `Agent 09: Telemetry & Hardware Roofline Profiler`
**Target Codebase Paths**:
- `neuralops/runners/monitor/`
- `neuralops/tracking/`
- `neuralops/analysis/`
- `neuralops/benchmarks/`

---

## 1. Executive Summary & Design Philosophy

Asynchronous GPU kernel execution presents fundamental measurement pitfalls when using CPU wall-clock timers (`time.time()`, `time.perf_counter()`). The CPU launch queue decouples kernel submission from hardware execution on the streaming multiprocessors (SMs), dynamic GPU boost clocks introduce frequency variance across runs, and warm L2 cache lines distort microbenchmarking results. Furthermore, isolated latency metrics fail to diagnose *why* a kernel is bottlenecked—whether execution is throttled by high-bandwidth memory (HBM/GDDR6) transfer rates or Tensor Core arithmetic throughput.

Part 09 documents the **Hardware Telemetry, Profiling & Roofline Subsystem** of NeuralOps:
1. **Asynchronous Microsecond Event Timing**: Eliminates host-device queue skew via native `torch.cuda.Event(enable_timing=True)`, explicit stream synchronization (`torch.cuda.synchronize()`), warmup loops, and outlier-resistant statistical reduction (trimmed mean, median, standard deviation).
2. **True Peak VRAM Instrumentation**: Accurately tracks high-water memory footprints via `torch.cuda.max_memory_allocated()` combined with cache purging (`torch.cuda.empty_cache()`, `torch.cuda.reset_peak_memory_stats()`), and non-intrusive NVML / `nvidia-smi` hardware sampling.
3. **Empirical SM 8.9 Roofline Engine**: Computes exact operational arithmetic intensity ($I = \text{FLOPs} / \text{Bytes}$) across all 131 attention mechanisms and 72 compression engines, mapping them relative to the $317.9\,\text{FLOPs/Byte}$ hardware ridge inflection point on NVIDIA Ada Lovelace architectures (RTX 4070 SM 8.9 AD104 silicon).
4. **Real-Time WebSocket & SSE Telemetry**: Exposes high-frequency streaming telemetry over `/ws/telemetry` via FastAPI and `ConnectionManager`, broadcasting live GPU metrics, speculative decoding stats, and paged KV cache allocation maps.
5. **Pluggable Experiment Tracking**: Provides a unified facade (`ExperimentTracker`) supporting local JSONL logging, TensorBoard, Weights & Biases, and MLflow backends.

---

## 2. Core Subsystem Architecture

```
                      ┌────────────────────────────────────────┐
                      │        NeuralOps Workload Target       │
                      │  (Attention Kernel / SVD Compression)  │
                      └───────────────────┬────────────────────┘
                                          │
                  ┌───────────────────────┴───────────────────────┐
                  ▼                                               ▼
   ┌─────────────────────────────┐                 ┌─────────────────────────────┐
   │ High-Precision CUDA Harness │                 │  Hardware Roofline Engine   │
   │  (torch.cuda.Event timing,  │                 │  (Analytical FLOPs & Bytes, │
   │   max_memory_allocated)     │                 │   Arithmetic Intensity I)   │
   └──────────────┬──────────────┘                 └──────────────┬──────────────┘
                  │                                               │
                  └───────────────────────┬───────────────────────┘
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    Live Telemetry & Dashboard Layer    │
                      │ (NVML / nvidia-smi / Monitor API / WS) │
                      └───────────────────┬────────────────────┘
                                          │
             ┌────────────────────────────┼────────────────────────────┐
             ▼                            ▼                            ▼
┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
│  FastAPI /ws/telemetry  │  │   SSE /api/events/live  │  │ ExperimentTracker Facade│
│  (ConnectionManager)    │  │   (Real-time Streams)   │  │ (Local/WandB/TensorBoard│
└─────────────────────────┘  └─────────────────────────┘  └─────────────────────────┘
```

---

## 3. Microsecond Timing Harness & Memory Profiling

### 3.1 Asynchronous CUDA Event Synchronization Protocol

Implemented in `neuralops/benchmarks/harness.py` (`BenchmarkRunner.measure_callable`) and `neuralops/analysis/roofline.py` (`profile_roofline`):

```python
# From neuralops/benchmarks/harness.py
def measure_callable(
    self,
    fn: Callable[[], Any],
    total_flops: float = 0.0,
    total_bytes: float = 0.0,
    batch_size: int = 1,
    seq_len: int = 1,
    warmup: Optional[int] = None,
    iters: Optional[int] = None,
) -> BenchmarkMetric:
    w_iters = warmup if warmup is not None else self.warmup_iters
    b_iters = iters if iters is not None else self.benchmark_iters

    self.reset_vram()

    # 1. Warmup Loop (stabilizes GPU clocks, compiles JIT kernels)
    for _ in range(w_iters):
        fn()

    if self.device.type == "cuda" and torch.cuda.is_available():
        # 2. Host-Device Stream Barrier
        torch.cuda.synchronize(self.device)
        self.reset_vram()

        # 3. Microsecond-Accurate Event Allocation
        start_events = [torch.cuda.Event(enable_timing=True) for _ in range(b_iters)]
        end_events = [torch.cuda.Event(enable_timing=True) for _ in range(b_iters)]

        # 4. Asynchronous Kernel Recording
        for i in range(b_iters):
            start_events[i].record()
            fn()
            end_events[i].record()

        # 5. Pipeline Barrier Synchronization
        torch.cuda.synchronize(self.device)
        latencies = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    else:
        # High-resolution host fallback
        latencies = []
        for _ in range(b_iters):
            t0 = time.perf_counter()
            fn()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)
```

### 3.2 Robust Statistical Metrics
To guarantee reproducible benchmarking unaffected by operating system interrupts or thermal throttling, latency samples are reduced into:
- **Trimmed Mean**: Excluding top and bottom 10% outliers:
  $$\mu_{10\%} = \frac{1}{N - 2k} \sum_{i=k+1}^{N-k} t_{(i)}$$
- **Median & Standard Deviation**: Asserting coefficient of variation:
  $$\frac{\sigma}{\mu} < 0.05 \quad (\text{less than } 5\% \text{ noise})$$

### 3.3 VRAM Instrumentation & High-Water Tracking
Implemented in `neuralops/benchmarks/harness.py` and `neuralops/runners/monitor/telemetry.py`:
- **Pre-execution cleanup**:
  ```python
  gc.collect()
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats(self.device)
  ```
- **Peak footprint capture**:
  ```python
  peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)
  ```
- **Real-time monitor sampling**: `query_hardware_gpu()` extracts active allocations (`torch.cuda.memory_allocated`), cached memory (`torch.cuda.memory_reserved`), and queries `nvidia-smi` for power draw ($W$), graphics clocks ($\text{MHz}$), and temperature ($^\circ C$).

---

## 4. SM 8.9 Hardware Roofline Model & Arithmetic Intensity

### 4.1 Theoretical & Operational Ceilings (NVIDIA RTX 4070 SM 8.9 AD104 Silicon)
The hardware limits of the primary physical testbed (NVIDIA GeForce RTX 4070, AD104 silicon: 5,888 CUDA cores, 46 SMs, 184 4th Gen Tensor Cores) are formalized in `neuralops/analysis/roofline.py` (`get_device_roofline_specs`):
- **Peak FP16 Tensor Core Compute**: $P_{\text{peak}} = 160.2\,\text{TFLOPs/s}$ (Boost) / $141.6\,\text{TFLOPs/s}$ (Base)
- **Peak DRAM Memory Bandwidth (GDDR6X)**: $B_{\text{peak}} = 504.0\,\text{GB/s}$
- **Hardware Ridge Point (Inflection Intensity)**:
  $$I^* = \frac{P_{\text{peak}}}{B_{\text{peak}}} = \frac{160.2 \times 10^{12}}{504.0 \times 10^9} \approx 317.9\,\text{FLOPs/Byte}$$
  *(On conservative or laptop thermal profiles with lower SM clock ceilings, $P_{\text{peak}}$ scales proportionally while GDDR6 bandwidth provides $256.0\,\text{GB/s}$ to $504.0\,\text{GB/s}$).*

### 4.2 Attainable Performance Formula
$$\text{Performance}(I) = \min\left(P_{\text{peak}}, \; I \cdot B_{\text{peak}}\right)$$

### 4.3 Analytical FLOP and Byte Formulations
Implemented in `neuralops/analysis/roofline.py` (`estimate_module_flops_and_bytes`) and `neuralops/analysis/flops_utils.py`:

| Mechanism Type | Core Forward FLOPs | Naive DRAM Traffic (Bytes) | SRAM-Fused DRAM Traffic (Bytes) |
|---|---|---|---|
| **Standard Softmax (Full)** | $4 B H T^2 d + 8 B T D^2$ | $4 \cdot \text{elem} \cdot B H T^2 + \text{Param} + \text{IO}$ | $\text{IO} + \text{Param} + \text{QKV}_{\text{boundaries}}$ |
| **Causal Softmax** | $2 B H T(T+1) d + 8 B T D^2$ | $2 \cdot \text{elem} \cdot B H T^2 + \text{Param} + \text{IO}$ | $\text{IO} + \text{Param} + \text{QKV}_{\text{boundaries}}$ |
| **Linear Recurrent (GDN / GLA / KDA)** | $4 B H T d^2 + 8 B T D^2$ | $2 \cdot \text{elem} \cdot B H T d^2 + \text{Param} + \text{IO}$ | $\text{IO} + \text{Param} + \text{QKV}_{\text{boundaries}}$ |
| **Sliding Window / Local** | $4 B H T \min(T, W) d + 8 B T D^2$ | $4 \cdot \text{elem} \cdot B H T \min(T, W) + \text{Param} + \text{IO}$ | $\text{IO} + \text{Param} + \text{QKV}_{\text{boundaries}}$ |

### 4.4 Bottleneck Regime Classification
Based on operational arithmetic intensity $I_{\text{fused}} = \text{FLOPs} / \text{Bytes}_{\text{fused}}$:
1. **Memory-Bandwidth Bound ($I < 0.85 \cdot I^*$)**: Kernels throttled by DRAM/GDDR6 transfer rates. Attention mechanisms in this regime benefit directly from SVD low-rank projection and KV-cache compression.
2. **SRAM-Fused Transition Regime ($I_{\text{fused}} > 1.5 \cdot I_{\text{naive}}$)**: Intermediate attention matrices are kept entirely in SRAM tiles (FlashAttention / Triton GDN chunked), mitigating the memory wall.
3. **Compute-Bound ($I \ge 0.85 \cdot I^*$)**: Tensor Cores fully saturated; throughput governed by peak arithmetic capacity.

---

## 5. Real-Time Telemetry & WebSocket Streaming Engine

### 5.1 Architecture of `ConnectionManager`
Implemented in `neuralops/runners/monitor/fastapi_app.py`:

```python
class ConnectionManager:
    """Manages active concurrent WebSocket client connections for real-time telemetry streaming."""
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.active_connections.discard(websocket)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        to_remove = set()
        for conn in list(self.active_connections):
            try:
                await conn.send_json(message)
            except (RuntimeError, IOError, ConnectionError, OSError):
                to_remove.add(conn)
        for dead in to_remove:
            self.active_connections.discard(dead)
```

### 5.2 High-Frequency Stream Payload
The `/ws/telemetry` endpoint emits structured JSON telemetry ticks every 1.0 second:
```json
{
  "event": "telemetry_tick",
  "gpu": {
    "gpu_name": "NVIDIA GeForce RTX 4070 Laptop GPU",
    "gpu_util": 94.2,
    "mem_used_mb": 3850.0,
    "mem_total_mb": 8188.0,
    "mem_reserved_mb": 4096.0,
    "power_w": 88.5,
    "temp_c": 68.0,
    "clock_mhz": 2175.0,
    "status": "online"
  },
  "speculative": {
    "acceptance_rate": 0.785,
    "mean_accepted_tokens": 3.92,
    "speedup_factor": 2.45,
    "gamma_curves": [...]
  },
  "paged_kv": {
    "utilization_pct": 32.8,
    "allocated_blocks": 21,
    "free_blocks": 43,
    "continuous_batching": {...}
  }
}
```

### 5.3 REST & SSE Endpoints
- `GET /ws/telemetry`: WebSocket stream with bidirectional ping/pong heartbeats.
- `GET /api/events/live`: Server-Sent Events (SSE) streaming fallback.
- `GET /api/gpu/metrics`: Instantaneous snapshot conforming to `GpuMetrics` Pydantic DTO.
- `GET /api/benchmarks/history`: Historical telemetry curves (VRAM, utilization, power, temperature).
- `GET /api/attentions`: Complete catalog of 131 attention mechanisms with latencies and complexity.
- `GET /api/compressors`: Pareto leaderboard of 72 compression engines.
- `GET /api/paged_kv_cache`: Virtual page table and block allocation state.
- `GET /api/speculative_decoding`: Acceptance rates and speedup curves.

---

## 6. Unified Experiment Tracking Facade

Implemented in `neuralops/tracking/tracker.py` and `neuralops/tracking/backends.py`:
- **Supported Backends**: `local` (JSONL format), `tensorboard`, `wandb`, and `mlflow`.
- **Zero-Dependency Local Logging**: Automatically logs structured metric records and hyperparameter JSON files without external service dependencies:
  ```python
  tracker = ExperimentTracker(backend="local", log_dir="./results/logs", experiment_name="gdn_sweep")
  tracker.log_metric("train/loss", 0.342, step=100)
  tracker.log_metric("hw/peak_vram_mb", 3850.0, step=100)
  tracker.finalize({"eval/accuracy": 91.63})
  ```

---

## 7. Audit & Verification Standards

### 7.1 Verified Test Suites
1. **Monitor Server & WebSocket Suite**: `tests/test_monitor_server.py`
   - `test_gpu_telemetry_fallback`: Verifies graceful offline fallback when CUDA is absent.
   - `test_advanced_monitor_endpoints`: Verifies discovery of all 131 attentions, 72 compressors, PagedKV, and Speculative Decoding states.
   - `test_websocket_multi_client_concurrent_stress`: Connects multiple concurrent WebSocket clients, verifies receipt of `"telemetry_tick"` JSON payload, and ensures zero connection leak in `ConnectionManager`.
   - `test_websocket_resiliency_rapid_reconnect`: Verifies rapid connect/disconnect and ping/pong heartbeats without server degradation.
2. **Attention Memory Profiler**: `tests/test_attention_profiler.py`
   - Verifies execution latency and peak memory tracking on attention layers.
3. **Experiment Tracking Regressions**: `tests/test_experiment_logging_tracker.py`
   - Verifies metric logging across backends and local file persistence.

### 7.2 Static Verification & Quality Gates
- `ruff check neuralops/runners/monitor neuralops/tracking neuralops/analysis/roofline.py neuralops/benchmarks/harness.py`: **0 errors / All checks passed**.
- No hardcoded metric fallbacks in data producers (strict compliance with `AGENTS.md` §6.2).
- Zero fake artifacts policy adhered to in export routines (`AGENTS.md` §6.1).
