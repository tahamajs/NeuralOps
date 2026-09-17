# Pre-Submission Checklist — AdaSplash

## Setup
- [ ] Copied `_TEMPLATE/` to `adasplash/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module adasplash`
- [ ] No `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] **Clearly states AdaSplash is NOT a compression method**
- [ ] α, W, S stated
- [ ] Sparsity % stated
- [ ] Parameter **increase** stated

### Introduction
- [ ] Motivation: sparse attention, not compression
- [ ] α-entmax family introduced
- [ ] Attention sinks and splash masking explained
- [ ] Contributions verifiable

### Method
- [ ] Algorithm 1 (entmax bisection) complete
- [ ] Algorithm 2 (forward) complete
- [ ] Complexity analysis
- [ ] Parameter count analysis (honest)

### Results
- [ ] Table 7.1 main results
- [ ] Table 7.3 sparsity
- [ ] Table 7.5 α ablation
- [ ] Table 7.6 splash radius
- [ ] Table 7.7 sinks
- [ ] Table 7.8 LoRA rank

### Analysis
- [ ] Sparsity distribution
- [ ] Attention interpretability
- [ ] **Inference cost breakdown** (why slower)
- [ ] Comparison with softmax/sparsemax

## Honesty (CRITICAL)

- [ ] **Never call AdaSplash "compression"**
- [ ] Parameter increase clearly stated
- [ ] Latency overhead explained
- [ ] Sparsity ≠ speedup (PyTorch dense matmuls)
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation stated

## Reproducibility

- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hyperparameters documented
- [ ] Entmax implementation described

## Final Pass

- [ ] Read aloud
- [ ] Verify all numbers
- [ ] All tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/adasplash
nano papers/adasplash/variables.json
cd papers/adasplash
python apply_vars.py --module adasplash
grep -rn "{{" .
python build.py --module adasplash --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE adasplash
```

### مرحله ۲: اجرا

```powershell
cd D:\SVD_Project
python run_all_4_adasplash.py --models distilbert,roberta,gpt2 --alpha 1.5 --splash_radius 32 --lora_rank 16 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\adasplash
python apply_vars.py --module adasplash
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module adasplash --pdf
