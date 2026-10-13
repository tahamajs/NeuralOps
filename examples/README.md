# NeuralOps Examples

Executable recipes showcasing the core capabilities of NeuralOps: zero-shot attention replacement, deep model compression, distillation fine-tuning, Hugging Face AutoModel integration, fast inference, and unified CLI sweeps.

## 📋 Examples Index

| File | What it demonstrates | Primary APIs |
| :--- | :--- | :--- |
| [`basic_attention.py`](basic_attention.py) | Zero-shot attention replacement in a pretrained transformer | `neuralops.replace_attention` |
| [`compression_demo.py`](compression_demo.py) | Low-rank model compression with parameter reduction | `neuralops.compress` |
| [`fine_tuning_demo.py`](fine_tuning_demo.py) | Mixed-precision Knowledge Distillation (KD) fine-tuning | `MixedPrecisionTrainer`, KD loss |
| [`hf_integration.py`](hf_integration.py) | Native Hugging Face `AutoConfig` and `AutoModel` generation | `AutoConfig`, `AutoModelForCausalLM` |
| [`inference_demo.py`](inference_demo.py) | Autoregressive decoding and generation using inference engine | `NeuralOpsInferenceEngine` |
| [`benchmark_sweep.py`](benchmark_sweep.py) | Running programmatically defined sweeps via Python API | `neuralops.benchmarks.Runner` |
| [`cli_usage.sh`](cli_usage.sh) | Executing sweeps, plans, and verification via the CLI | `neuralops.benchmarks.cli` |

---

## 🚀 Running Any Example

Every example is standalone, verified, and executes on CPU, MPS, or CUDA:

```bash
# Attention replacement
python examples/basic_attention.py

# Low-rank model compression
python examples/compression_demo.py

# Knowledge distillation training
python examples/fine_tuning_demo.py

# Hugging Face AutoModel generation
python examples/hf_integration.py

# Fast inference engine
python examples/inference_demo.py

# Benchmark sweep
python examples/benchmark_sweep.py

# Command-line suite
bash examples/cli_usage.sh
```
