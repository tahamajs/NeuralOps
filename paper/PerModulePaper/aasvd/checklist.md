# Pre-Submission Checklist — AASVD

## Setup
- [ ] Copied `_TEMPLATE/` to `aasvd/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module aasvd`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Real parameter reduction stated
- [ ] Accuracy drop stated
- [ ] Calibration approach mentioned

## Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal
- [ ] Contributions verifiable
- [ ] Organization complete

## Method
- [ ] All formulas numbered
- [ ] All formulas referenced
- [ ] Algorithm pseudocode complete
- [ ] Complexity analysis

## Results
- [ ] All tables filled with real numbers
- [ ] Rank ablation
- [ ] Covariance ablation
- [ ] EMA β ablation
- [ ] Calibration source ablation

## Honesty
- [ ] Parameter reduction verified (params_after < params_before)
- [ ] Accuracy delta honest
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Calibration corpus described

## Final Pass
- [ ] Read aloud
- [ ] Verified all numbers
- [ ] All tables render
- [ ] PDF compiles
- [ ] External feedback obtained

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/aasvd
nano papers/aasvd/variables.json
cd papers/aasvd
python apply_vars.py --module aasvd
grep -rn "{{" .  # verify empty
python build.py --module aasvd --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE aasvd
```

### مرحله ۲: اجرای AASVD

```powershell
cd D:\SVD_Project
python run_all_4_aasvd.py --models distilbert,roberta,gpt2 --rank 288 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_aasvd/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\aasvd
python apply_vars.py --module aasvd
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module aasvd
