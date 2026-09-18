# Phase 0: Pre-Run Verification Report

## Compressor Arguments & Invocation

To ensure a perfectly fair comparison, we have audited the `__init__` and registration mechanisms for the 6 target compressors in the framework. All compressors accept dynamic `**kwargs` from the `run_quality_sweep.py` parser, ensuring precise control.

Here are the confirmed arguments for each target:

1.  **`kmeans`** (`compression_engine/kmeans_quantize.py`)
    *   **Arguments:** `--kmeans_clusters 256 --kmeans_lora_r 16`
    *   **Mechanism:** Intercepted by `kmeans_factory()`, passed down to `KMeansCompressor()`. Uses LoRA fine-tuning for adapters.

2.  **`hybrid_svd_layer`** (`compression_engine/hybrid_svd_layer.py`)
    *   **Arguments:** `--hybrid_svd_rank 128 --hybrid_svd_lora_r 16`
    *   **Mechanism:** Standard compressor class. Combines PCA/SVD with an optional dense fallback and LoRA.

3.  **`come`** (`compression_engine/come.py`)
    *   **Arguments:** `--come_sparsity 0.3`
    *   **Mechanism:** Uses `CoMeCompressor()`. Expects sparsity target, dynamically allocates rank.

4.  **`sdrm`** (`compression_engine/sdrm_compressor.py`)
    *   **Arguments:** `--total_param_ratio 0.7 --lora_r 16`
    *   **Mechanism:** `SDRMCompressor` defaults to `0.7` ratio and fallback, but we enforce identical adapter ranks via `--lora_r 16`. 

5.  **`a3`** (`compression_engine/a3_compressor.py`)
    *   **Arguments:** `--a3_max_error 0.5`
    *   **Mechanism:** Direct truncation based on cumulative sum of singular values.

6.  **`adaptive_svd_attention`** (`attentions/adaptive_svd_attention.py`)
    *   **Arguments:** None required locally (handles internal thresholding based on base `config`).
    *   **Mechanism:** Injected dynamically into the attention registry (`ATTENTION_REGISTRY`), whereas the other five are in `COMPRESSOR_REGISTRY`. 

## Dry Run Results
A local python testing script (`dry_run_compressors.py`) was executed on the remote Windows server (`hadi-windows`) targeting a local dummy `distilbert_sst2` model. The script verified that each of the compressors can be safely instantiated and queried without crashing or failing to register.

## Environment Resources
We verified that the host system (`hadi-windows`) has exactly **8GB of VRAM (RTX 4070)**. The environment currently has `5825 MB` of Free VRAM while executing the previous 92-module sweep. 
We have generated `run_fair_comparison.py`, which is hardcoded to enforce sequential execution (1 worker) and `batch_size=32`, with explicitly enforced memory cleaning (`torch.cuda.empty_cache()`) between runs, to guarantee zero chance of OOM.

## Conclusion
All pre-flight checks are complete and the evaluation script is ready on the server. The actual fair comparison sweep will be initiated *after* the current parallel sweep finishes to avoid contention.
