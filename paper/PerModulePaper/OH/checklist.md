# Pre-Submission Checklist — OHA

## Setup
- [ ] Copied `_TEMPLATE/` to `oha/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module oha`
- [ ] Verified no `{{PLACEHOLDER}}` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Parameter increase stated**
- [ ] **"Compression" clarified** (sequence, not model)
- [ ] Accuracy drop stated

### Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal
- [ ] Contributions verifiable
- [ ] **Explicit statement that this is not a compressor**

### Method
- [ ] All formulas numbered
- [ ] Compression mechanism detailed
- [ ] Gate initialization explained
- [ ] Complexity analysis
- [ ] Parameter analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows increase**
- [ ] Compression rank ablation
- [ ] Window size ablation
- [ ] Gate bias ablation
- [ ] LoRA rank ablation

### Analysis
- [ ] Gate evolution
- [ ] Per-head analysis
- [ ] Compression quality
- [ ] Comparison with Infini
- [ ] `raise` problem discussed

### Limitations
- [ ] **Not a compressor**
- [ ] Cross-attention not supported
- [ ] `raise` on copy failure
- [ ] Task coverage
- [ ] Single-seed

## Honesty
- [ ] Parameter increase admitted
- [ ] "Compression" clarified (sequence length)
- [ ] Failure cases documented
- [ ] Negative results in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/oha
nano papers/oha/variables.json
cd papers/oha
python apply_vars.py --module oha
grep -rn "{{" .
python build.py --module oha --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE oha
```

### مرحله ۲: اجرای OHA

```powershell
cd D:\SVD_Project
python run_all_4_oha.py --models distilbert,roberta,gpt2 --window_size 64 --compression_rank 32 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\oha
python apply_vars.py --module oha
python build.py --module oha
