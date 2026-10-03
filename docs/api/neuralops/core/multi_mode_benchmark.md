Module neuralops.core.multi_mode_benchmark
==========================================
Unified Multi-Mode Evaluation Runner for Efficient Transformer Toolkit.

Evaluates transformer architectures across 10+ next-generation inference & execution modes:
1. `dense_fp16`: Baseline standard attention in FP16/BF16
2. `flash_attention_3_fp8`: Hopper FP8 WGMMA accelerated exact tiled attention
3. `svd_linear_lowrank`: SVD low-rank factorized attention
4. `rotational_slicegpt`: Orthogonal rotational invariance + sliced channels
5. `sparsegpt_int4`: Astro INT4 / 2:4 structured pruned execution
6. `paged_deltakv`: Paged Delta-KV temporal difference cache compression
7. `mamba_ssm_scan`: Selective SSM chunked associative scan
8. `moe_routing`: Mixture of Block Attention (MoBA)
9. `aegis_spectron_hybrid`: Multi-branch spectral entropy dynamic routing
10. `kronecker_tt_adapter`: Kronecker Tensor-Train PEFT adapter
11. `kmeans_quantized`: Clustered codebook quantization
12. `nystrom_kernel`: Nystrom landmark kernel approximation

Measures:
- Perplexity (PPL)
- Peak VRAM / Memory Footprint (MB)
- Latency (ms/token and ms/step)
- Throughput (tokens/sec)
- Compression Ratio (x factor)
- Memory Savings (%)

Provides:
- `MultiModeBenchmarkRunner` class
- Flexible multi-dataset loading (GLUE, WikiText-2, C4, Multi-Choice, Synthetic)
- CLI entrypoint with comprehensive reporting & JSON / Pandas DataFrame / CSV export

Functions
---------

`main()`
:   

`parse_args()`
:   

Classes
-------

`ModeBenchmarkResult(mode: str, model_name: str, dataset: str, device: str, dtype: str, batch_size: int, seq_len: int, num_samples: int, perplexity: float, loss: float, peak_vram_mb: float, allocated_vram_mb: float, latency_ms_per_token: float, latency_ms_per_step: float, throughput_tokens_per_sec: float, num_parameters: int, compression_ratio: float, memory_savings_pct: float, execution_status: str = 'SUCCESS', error_message: str | None = None, extra_metrics: dict[str, typing.Any] = <factory>)`
:   Benchmark outcome for a single execution mode.

    ### Instance variables

    `allocated_vram_mb: float`
    :   The type of the None singleton.

    `batch_size: int`
    :   The type of the None singleton.

    `compression_ratio: float`
    :   The type of the None singleton.

    `dataset: str`
    :   The type of the None singleton.

    `device: str`
    :   The type of the None singleton.

    `dtype: str`
    :   The type of the None singleton.

    `error_message: str | None`
    :   The type of the None singleton.

    `execution_status: str`
    :   The type of the None singleton.

    `extra_metrics: dict[str, typing.Any]`
    :   The type of the None singleton.

    `latency_ms_per_step: float`
    :   The type of the None singleton.

    `latency_ms_per_token: float`
    :   The type of the None singleton.

    `loss: float`
    :   The type of the None singleton.

    `memory_savings_pct: float`
    :   The type of the None singleton.

    `mode: str`
    :   The type of the None singleton.

    `model_name: str`
    :   The type of the None singleton.

    `num_parameters: int`
    :   The type of the None singleton.

    `num_samples: int`
    :   The type of the None singleton.

    `peak_vram_mb: float`
    :   The type of the None singleton.

    `perplexity: float`
    :   The type of the None singleton.

    `seq_len: int`
    :   The type of the None singleton.

    `throughput_tokens_per_sec: float`
    :   The type of the None singleton.

    ### Methods

    `to_dict(self) ‑> dict[str, typing.Any]`
    :

`ModeTransformEngine()`
:   Transforms a baseline PyTorch / HF model or attention module into one of
    the 10+ target execution modes.

    ### Static methods

    `apply_mode(model: torch.nn.modules.module.Module, mode: str, device: torch.device, dtype: torch.dtype = torch.float16, rank: int = 32, sparsity_ratio: float = 0.5) ‑> torch.nn.modules.module.Module`
    :   Apply the specified mode transformation in-place or via wrapper.

`MultiModeBenchmarkRunner(model_name_or_path: str = 'distilbert-base-uncased', device: str | None = 'auto', dtype: str = 'float16', batch_size: int = 8, seq_len: int = 128, num_samples: int = 128, warmup_steps: int = 5, benchmark_steps: int = 15, output_dir: str = './benchmark_results', allow_synthetic: bool = False)`
:   Unified Multi-Mode Evaluation Runner.
    
    Executes systematic benchmarks across 12 distinct inference modes, measuring:
    - Perplexity (PPL)
    - Peak VRAM / Memory footprint
    - Latency (ms/token, ms/step)
    - Throughput (tokens/sec)
    - Compression Ratio & Memory Savings (%)

    ### Static methods

    `list_supported_modes() ‑> list[str]`
    :   Return all supported benchmark execution modes.

    ### Methods

    `benchmark_attention_module(self, attention_type: str, hidden_size: int = 64, num_heads: int = 4, seq_len: int = 64, batch_size: int = 2, num_iters: int = 10, **kwargs: Any) ‑> dict[str, typing.Any]`
    :   Directly benchmark a registered attention module forward pass.

    `export_csv(self, filepath: str | pathlib.Path | None = None) ‑> str`
    :   Export results to CSV file.

    `export_json(self, filepath: str | pathlib.Path | None = None) ‑> str`
    :   Export all benchmark outcomes to structured JSON.

    `print_summary_table(self)`
    :   Print an aligned ASCII summary table of the benchmark results.

    `run_all_modes(self, modes: list[str] | None = None, datasets: list[str] | None = None, rank: int = 32) ‑> list[neuralops.core.multi_mode_benchmark.ModeBenchmarkResult]`
    :   Execute evaluation sweep across all requested modes and datasets.

    `run_single_mode(self, mode: str, dataset_name: str = 'sst2', rank: int = 32) ‑> neuralops.core.multi_mode_benchmark.ModeBenchmarkResult`
    :   Run benchmark for a single execution mode.

    `to_dataframe(self) ‑> Any`
    :   Convert results to a pandas DataFrame.

`SyntheticTextDataset(vocab_size: int = 30522, seq_len: int = 128, num_samples: int = 128)`
:   Synthetic dataset for benchmarking token throughput and perplexity.

    ### Ancestors (in MRO)

    * torch.utils.data.dataset.Dataset
    * typing.Generic