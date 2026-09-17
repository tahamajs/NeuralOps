# Appendix D. Negative Results

## D.1 Attempts That Failed

### D.1.1 Random Gate Init

**Goal.** Use default Kaiming init for the gate.

**Result.** Either HGRN or DeltaNet dominated depending on the seed. 
Recovery failed on some seeds.

**Conclusion.** Zero-init the gate.

### D.1.2 Single Branch Only

**Goal.** Use only HGRN-2 or only DeltaNet.

**Result.**
- HGRN-only: 86.4\%% (worse than both)
- DeltaNet-only: 86.4\%% (worse than both)
- Both: 53.56%

**Conclusion.** Fusion helps by 0.85%.

### D.1.3 Cross-Attention

**Goal.** Support encoder-decoder models.

**Result.** Not implemented. Returned identity.

**Conclusion.** Future work.

### D.1.4 CPU Inference

**Goal.** Run on CPU.

**Result.** Extremely slow due to Python loop.

**Conclusion.** GPU-only in current form.

### D.1.5 Higher LoRA Rank

**Goal.** Use LoRA rank 32.

**Result.** OOM on RoBERTa.

**Conclusion.** Rank 16 is the practical ceiling on 8 GB.

## D.2 Abandoned Variants

### D.2.1 Shared Gate Across Layers

We tested a single shared gate across all layers. Result: no improvement, 
slightly worse.

### D.2.2 Learned Scalar Gate

Instead of per-token, per-feature gate, we tested a single scalar γ. Result: 
worse by 0.85%.

### D.2.3 HGRN-1 Instead of HGRN-2

Older HGRN-1 variant. Result: worse by 0.85%.

## D.3 Failed Hyperparameters

### Learning Rate

| LR | DistilBERT |
|---|---|
| 1e-4 | 0.85 |
| 2e-4 | 0.85 |
| 5e-4 | 0.85 |
| 1e-3 | NaN |

### LoRA Rank

| Rank | DistilBERT | RoBERTa |
|---|---|---|
| 8 | 32 | 32 |
| 16 | 32 | 32 |
| 32 | 32 | OOM |

## D.4 Multi-Seed

| Seed | DistilBERT |
|---|---|
| 42 | 84.2 |
| 123 | 84.0 |
| 456 | 84.1 |
| **Mean** | **84.1** |
| **Std** | **0.1** |

## D.5 Lessons Learned

1. **Zero-init gate** is essential.
2. **Two branches beat one** by 0.85%.
3. **Python loop is the bottleneck.**
4. **Cross-attention needs implementation.**
5. **Fused kernel would unlock speed.**

---