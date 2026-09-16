<!-- ═══════════════════════════════════════════════════════════════════════════
     FILE: 10_limitations.md
     PURPOSE: Honest limitations
     DEPENDENCIES: 09_discussion.md
     LENGTH TARGET: 0.5 - 1 page
     ═══════════════════════════════════════════════════════════════════════════ -->

# 10. Limitations

## 10.1 Inference Memory Unchanged

**This is the most significant limitation.** Our forward pass reconstructs 
the full weight matrix W before applying it to the input. As a result, 
inference VRAM is identical to the baseline.

**Fix.** Implement the fused forward pass:

```python
z = F.linear(x, self.Vt)
z = z * (self.S * self.gamma)
y_svd = F.linear(z, self.U)
y_lora = F.linear(F.linear(x, self.B), self.A)
return y_svd + y_lora + self.bias
```

Expected speedup: 2-4×.

## 10.2 `raise RuntimeError` in `copy_weights`

Unlike other modules in our framework, SALT **raises** on extraction failure:

```python
except Exception as e:
    raise RuntimeError(f"copy_weights failed: {e}")
```

This aborts injection for the entire model. Should be a warning.

## 10.3 `raise RuntimeError` on Zero Norm

Same issue for Frobenius norm check.

## 10.4 `output_scale` Unused

The `SALTLinear.__init__` defines `output_scale` but never uses it. Dead code.

## 10.5 Single Rank for SVD and Residual

We use the same $r$ for SVD truncation and residual rank. These could be 
independent.

## 10.6 GQA Pooling is Lossy

For Qwen with $H_{kv} < H$, we average KV groups. This loses information.

## 10.7 Task Coverage

Only SST-2. Behavior on generation, multi-class, or token-level tasks is 
untested.

## 10.8 Model Coverage

Only DistilBERT, RoBERTa, GPT-2. Qwen3.5 not completed.

## 10.9 Single Seed

No multi-seed runs. Differences below 3.4% may be noise.

## 10.10 No Fused Kernel

We do not provide a Triton or CUDA implementation of the fused forward.

## 10.11 No Comparison with PiSSA or OLoRA

We do not run head-to-head against related methods.

## 10.12 No Interaction with Quantization

The natural combination SALT + INT8 is untested.

## 10.13 Code Quality

Research-grade:
- Raising exceptions instead of warnings
- Dead `output_scale` code
- No unit tests for weight extraction

## 10.14 What This Paper Is Not

This is **not**:
- An inference-optimization method
- A state-of-the-art compression method
- A fused-kernel paper

---