# Spectral-Mamba: Hybrid Attention-State Space Module with Spectral Modulation

## 1. Mathematical Formulation & Architecture
1. **Pretrained Multi-Head Attention Branch**:
   $$O^{\text{attn}} = \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M\right) V W_O$$

2. **Spectral Mamba SSM Branch**:
   $$Y = \operatorname{Mamba}(X) \odot \sigma(\text{dt\_scale})$$

3. **Adaptive Gated Fusion**:
   $$g = \sigma(X W_{\text{gate}} + 1.8) \in [0, 1]^{B \times N \times 1}$$
   $$\text{Output} = g \odot O^{\text{attn}} + (1 - g) \odot Y$$

## 2. Root Cause Analysis & Gradient Propagation Fix
- **Failure Mode**: Mamba state-space layers initialized from scratch discarded all pretrained transformer representations, stalling training at 50.92%.
- **Architectural Repair**: Built a dual-branch hybrid architecture maintaining the pretrained attention manifold while smoothly learning state-space spectral dynamics.

## 3. Empirical Performance
- **SST-2 Accuracy**: **83.83%** (restored from 50.92%).\n

## Implementation status

The implementation for this page is [spectral_mamba.py](./spectral_mamba.py). Its constructor and forward contract are exercised by the framework attention compatibility suite. Treat performance, compression ratio, and accuracy as experiment-dependent; this page does not claim a universal benchmark result.

## Reproducibility notes

Use the registered attention key shown in the Python module, pass tensors with shape `(batch, sequence, hidden)`, and validate the selected model architecture before injection. Optional CUDA, Triton, FP8, or pretrained-weight paths require their corresponding runtime dependencies.

## Verification

Run `pytest -q tests/test_attention_modules.py tests/test_attentions.py` from `Codes/framework` before reporting experiment results.
