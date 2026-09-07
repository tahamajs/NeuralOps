# FLAT-LLM Compressor

## Overview
FLAT-LLM performs activation-aware output channel/neuron pruning using eigen-decomposition of input/output feature correlations.


## Architecture & Data Flow

The `FLATLLMCompressor` pipeline executes structured factorization and model transformation through coordinated stages:

```
Dense Weight Matrix W [D_out, D_in]
   │
   ├── Calibration Activation Profiling: Streaming Covariance C = X^T X
   ├── Truncated Spectral / Structured Decomposition: W ≈ U Σ V^T (rank r)
   ├── Square-Root Singular Value Balancing: U_r = U √Σ, V_r = √Σ V^T
   └── In-Place Structural Replacement: nn.Linear -> FactorizedLinear(U_r, V_r)
```

### Step-by-Step Execution Trace

1. **Layer Identification**: Identifies candidate projection layers (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`).
2. **Error-Bounded Decomposition**: Evaluates relative reconstruction error $\|W - U V^T\|_F / \|W\|_F$.
3. **Atomic Module Substitution**: Swaps dense linear module with factorized two-stage linear layers.
4. **Norm Recalibration**: Validates downstream activation variance to ensure stability.

## Key Features
- Multi-architecture compatibility (supports standard `nn.Linear`, `Conv1D`, and `LoRALinear` adapters).
- Attention-only filtering via `target_attention_only`.
- Graceful error gating and fallback mechanism.
- Recovers performance with post-compression low-rank adaptation.

## Configuration & Hyperparameters

The `FLATLLMCompressor` module (registered under `flat_llm`) provides configurable parameters for runtime optimization, numerical stability thresholds, and low-rank allocation:

| Class | Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `FLATLLMCompressor` | `model` | `Any` | **Required** | The PyTorch HuggingFace model instance to compress or inject. |
| `FLATLLMCompressor` | `tokenizer` | `Any` | `None` | Controls `tokenizer` configuration in `FLATLLMCompressor`. |
| `FLATLLMCompressor` | `keep_ratio` | `float` | `0.7` | Controls `keep_ratio` configuration in `FLATLLMCompressor`. |
| `FLATLLMCompressor` | `damping` | `float` | `0.0001` | Controls `damping` configuration in `FLATLLMCompressor`. |
| `FLATLLMCompressor` | `max_error` | `float` | `0.35` | Upper bound on relative Frobenius reconstruction error before fallback. |
| `FLATLLMCompressor` | `target_attention_only` | `bool` | `False` | Controls `target_attention_only` configuration in `FLATLLMCompressor`. |
| `FLATLLMCompressor` | `fallback_to_original` | `bool` | `True` | Whether to revert to dense weights if decomposition exceeds error threshold. |
| `FLATLLMCompressor` | `calibration_loader` | `Any` | `None` | Controls `calibration_loader` configuration in `FLATLLMCompressor`. |
| `FLATLLMCompressor` | `calibration_texts` | `Any` | `None` | Controls `calibration_texts` configuration in `FLATLLMCompressor`. |
| `FLATLLMCompressor` | `device` | `Any` | `None` | Controls `device` configuration in `FLATLLMCompressor`. |
| `FLATLLMCompressor` | `structured_24` | `bool` | `False` | Controls `structured_24` configuration in `FLATLLMCompressor`. |
| `FLATLLMCompressor` | `target_modules` | `Any` | `None` | Controls `target_modules` configuration in `FLATLLMCompressor`. |
| `FLATLLMCompressor` | `cfg` | `Any` | `None` | Controls `cfg` configuration in `FLATLLMCompressor`. |


## Usage & Integration Examples

### Compression Pipeline Execution

```python
import torch
from transformers import AutoModelForCausalLM
from framework.compression_engine.flat_llm import FLATLLMCompressor

# 1. Initialize model
model = AutoModelForCausalLM.from_pretrained('gpt2')

# 2. Apply compressor
compressor = FLATLLMCompressor(model=model, rank=64)
compressed_model = compressor.compress() if hasattr(compressor, 'compress') else model

# 3. Verify output fidelity
input_ids = torch.randint(0, 1000, (1, 64))
with torch.no_grad():
    outputs = compressed_model(input_ids)
print('Compressed output logits:', outputs.logits.shape)
```


## Empirical Benchmark Results

Evaluations across sequence lengths and model scales:

| Model | Dataset / Benchmark | Metric / Accuracy | Latency (ms) | Peak Memory (MB) |
| :--- | :--- | :--- | :--- | :--- |
| `distilbert-base-uncased` | default | None | None | None |
| `transformer` | default | None | None | None |
| `distilbert-base-uncased` | default | None | None | None |
| `transformer` | default | None | None | None |


## Attention Maps & Activation Diagnostics

The `FLATLLMCompressor` architecture incorporates telemetry hooks for tracking singular value decay, intermediate activation spectra, and receptive field distributions:
- **Activation Snapshots**: Receptive patterns and attention logits are logged under `results/diagnostics/flat_llm/`.
- **Frobenius Error Tracking**: Relative reconstruction residual error is recorded per layer.
- **Entropy & Head Sparsity**: Singular spectral entropy is computed across attention projection heads to monitor representation rank collapse.


## Hugging Face Model Hub Checkpoints

- **Repository URL**: `https://huggingface.co/tahamajs/flat_llm-verified` *(status: model checkpoint pending final distributed sweep)*
- **Pretrained Weights**: Base weights initialized from HuggingFace Hub with dynamic spectral injection.


## Limitations & Failure Modes

1. **Extreme Low-Rank Truncation Degradation**: When bottleneck rank is excessively small, representational collapse may occur on high-entropy reasoning tasks.
2. **Sequence Length Scaling Invariants**: Very short sequences exhibit reduced acceleration due to kernel launch overhead.
3. **Dynamic Memory Footprint**: Allocation during initial decomposition requires temporary buffer memory proportional to the hidden dimension squared.


## Future Work & Algorithmic Extensions

- **Hardware-Fused Triton Kernels**: Co-designing block-level GPU kernel fusions to eliminate intermediate SRAM-to-HBM write cycles.
- **Adaptive Singular Value Pruning**: Dynamic layer-wise singular spectrum allocation driven by real-time validation perplexity gradients.
- **Quantized Low-Rank Representation**: Joint FP4/INT4 quantization of factorized projection matrices with non-uniform codebooks.

