# Pre-Submission Checklist — EAT

## Setup
- [ ] Copied `_TEMPLATE/` to `eat/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module eat`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] **Parameter increase stated** (EAT is not a compressor)
- [ ] Accuracy drop stated
- [ ] Pruning mentioned (if used)

### Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal
- [ ] Contributions verifiable
- [ ] **Explicit statement that EAT is not a compressor**

### Method
- [ ] All formulas numbered
- [ ] All formulas referenced
- [ ] Algorithm pseudocode complete
- [ ] **Silent failure documented**

### Results
- [ ] Main results table complete
- [ ] **Parameter analysis shows increase**
- [ ] LoRA rank ablation
- [ ] Pruning ablation
- [ ] Recovery steps ablation

### Analysis
- [ ] Why it works on DistilBERT
- [ ] **Why it fails on RoBERTa**
- [ ] Inference cost breakdown
- [ ] Silent failure analysis

### Limitations
- [ ] **Not a compressor**
- [ ] Inference overhead
- [ ] Silent weight transfer failure
- [ ] Task coverage
- [ ] Single-seed

## Honesty
- [ ] Parameter increase admitted
- [ ] Failure cases reported
- [ ] Negative results in Appendix D
- [ ] Silent failure documented

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] All tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/eat
nano papers/eat/variables.json
cd papers/eat
python apply_vars.py --module eat
grep -rn "{{" .
python build.py --module eat --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE eat
```

### مرحله ۲: اجرای EAT

```powershell
cd D:\SVD_Project
python run_all_4_eat.py --models distilbert,roberta,gpt2 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_eat/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\eat
python apply_vars.py --module eat
python build.py --module eat
```

---

## 💡 نکات مهم EAT

### ۱. EAT **فشرده‌ساز نیست**

این مهم‌ترین نکته است. EAT پارامتر **اضافه** می‌کند:
- DistilBERT: +594,438 پارامتر (+0.89%)
- GPT-2: +1,188,876 پارامتر (+0.96%)
- RoBERTa: مشابه

**در همه جا این را صادقانه بگو.** در abstract، در intro، در results.

### ۲. مزیت اصلی = Progressive Token Pruning

اگر prune_layer فعال باشد:
- سرعت inference بهتر
- VRAM کمتر
- دقت کمی افت

**ولی فقط در inference**. در training غیرفعال است.

### ۳. Silent Failure = بزرگ‌ترین خطر

EAT در `copy_weights`:
```python
if missing_after:
    logger.warning(...)
    # ← ادامه می‌دهد با random init!
```

این باعث می‌شود RoBERTa collapse شود (50.92%).

**راه‌حل:** sanity check بعد از injection.

### ۴. RoBERTa مشکل‌دار است

از لاگ‌های واقعی:
- DistilBERT: PASS 90.25%
- RoBERTa: **FAIL 0.00%** (CUBLAS/OOM)
- GPT-2: PASS 78.78%

پس RoBERTa در مقاله باید با صداقت گزارش شود.

### ۵. `rank=64` استفاده نمی‌شود

`rank` پارامتر از `BaseAttention` است ولی EAT از آن برای compression استفاده نمی‌کند. فقط LoRA rank مهم است.

---

**قدم بعدی: `run_all_4_eat.py` را اجرا کن و نتایج `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**