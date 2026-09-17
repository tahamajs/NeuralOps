# 6. Experimental Setup

## 6.1 Benchmark Models and Hardware
Evaluations are conducted on standard transformer backbones across four model families:
- **DistilBERT** ($L=6, d=768, H=12$)
- **RoBERTa-base** ($L=12, d=768, H=12$)
- **GPT-2** ($L=12, d=768, H=12$)
- **Qwen 2.5 / 3.5** ($d=896$, GQA architecture)

Experiments are run on NVIDIA CUDA hardware with FP32 covariance accumulation and mixed-precision evaluation.

## 6.2 Calibration Protocol
Calibration utilizes multi-domain prompt sequences covering technical prose, dialogue, code snippets, and mathematical formulations (>512 tokens) to prevent domain-specific subspace collapse.
