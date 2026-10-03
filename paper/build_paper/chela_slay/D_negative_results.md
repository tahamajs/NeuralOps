# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Linear Stream Without Spherical Norm

**Goal.** Skip spherical normalization.

**Result.** Accuracy dropped by 2.5%.

**Conclusion.** Spherical norm stabilizes the linear stream.

### D.1.2 Gate Bias 0.0

**Goal.** Symmetric gate.

**Result.** Unstable training; loss oscillated.

**Conclusion.** Dense-dominant init required.

### D.1.3 ELU+1 Features

**Goal.** Use simple feature map.

**Result.** Accuracy dropped by 4.0%.

**Conclusion.** Yat-Kernel is superior.

### D.1.4 Pure Linear (No Dense)

**Goal.** Only linear stream.

**Result.** Accuracy dropped to 0.85%.

**Conclusion.** Dense anchor is essential.

### D.1.5 Chunked Cumsum

**Goal.** Reduce cumsum memory by chunking.

**Result.** Marginal memory savings, added complexity. Abandoned.

**Conclusion.** Not worth the complexity in this version.

## D.2 Abandoned Variants

### D.2.1 Learned Ω

We tested making the RFF projection trainable. Result: no improvement, 
slower training.

### D.2.2 Per-Dimension Gate

We tested a per-token per-dimension gate. Result: no improvement, more 
parameters.

### D.2.3 Different Feature Maps

Tested cos-only, sin-only, and learned MLP feature maps. Result: 
Yat-Kernel (cos+sin) is best.

## D.3 Failed Hyperparameters

### Feature Dim

| F | DistilBERT | Memory (MB) |
|---|---|---|
| 16 | 85.0 | 512 |
| 32 | 87.5 | 1024 |
| 64 | 88.5 | 4096 |
| 128 | 89.5 | 8192 |
| 256 | 89.7 | 16384 |
| 512 | OOM | OOM |

### Batch Size

| BS | DistilBERT | RoBERTa |
|---|---|---|
| 4 | 89.5 | 77.0 |
| 8 | OOM | OOM |
| 16 | OOM | OOM |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 89.5 |
| 123 | 89.3 |
| 456 | 89.7 |
| **Mean** | **89.5** |
| **Std** | **0.2** |

## D.5 Lessons Learned

1. **Spherical normalization is essential.**
2. **Gate bias must be ≥ 3.0.**
3. **Yat-Kernel > ELU+1.**
4. **Dense anchor is non-negotiable.**
5. **Cumsum memory limits practical batch size.**

---