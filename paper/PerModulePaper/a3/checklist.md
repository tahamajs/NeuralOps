# Pre-Submission Checklist — A3

## Setup
- [ ] Copied `_TEMPLATE/` to `a3/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module a3`
- [ ] Verified no `{{PLACEHOLDER}}` remains
- [ ] Built `paper_full.md`

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] All numbers match Section 7
- [ ] Parameter reduction stated
- [ ] Accuracy drop stated
- [ ] Any limitation mentioned

### Introduction
- [ ] Motivation is concrete
- [ ] Problem statement formal
- [ ] Contributions verifiable
- [ ] Organization complete

### Method
- [ ] All formulas numbered
- [ ] All formulas referenced in text
- [ ] Algorithms written as pseudocode
- [ ] Complexity analysis complete

### Results
- [ ] All tables filled
- [ ] All ablations present
- [ ] Training curves referenced
- [ ] Parameter counts verified

## Honesty
- [ ] Parameter reduction is real (verified)
- [ ] Accuracy delta is honest
- [ ] No overclaiming
- [ ] Failure cases documented in Appendix D
- [ ] Single-seed limitation stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HuggingFace
- [ ] Hardware specified
- [ ] Seeds documented
- [ ] Runtime estimates provided
- [ ] Calibration data described

## Appendices
- [ ] A: Mathematical proofs complete
- [ ] B: Implementation details sufficient
- [ ] C: Full data tables present
- [ ] D: Negative results documented
- [ ] E: Ethics discussed

## Final Pass
- [ ] Read entire paper aloud
- [ ] Check every number against source data
- [ ] Verify all tables render
- [ ] Verify all equations render
- [ ] Build PDF successfully
- [ ] Get 2+ external readers
- [ ] Address feedback

## Submit
- [ ] Choose venue
- [ ] Format according to guidelines
- [ ] Prepare supplementary
- [ ] Upload to arXiv
- [ ] Announce
- [ ] Update GitHub README with link

---

## Quick Start

```bash
# 1. Copy template
cp -r papers/_TEMPLATE papers/a3

# 2. Fill variables
nano papers/a3/variables.json

# 3. Apply
cd papers/a3
python apply_vars.py --module a3

# 4. Verify
grep -rn "{{" .  # should be empty

# 5. Fill manual sections (Method, Results, Analysis)

# 6. Build
python build.py --module a3 --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی تمپلیت

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE a3
```

### مرحله ۲: اجرای A3 و دریافت اعداد واقعی

```powershell
cd D:\SVD_Project
python run_all_4_a3.py --models distilbert,roberta,gpt2 --rank 64 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_a3/*/summary.csv` اعداد را بردار و در `variables.json` بگذار.

### مرحله ۴: Apply variables

```powershell
cd D:\SVD_Project\papers\a3
python apply_vars.py --module a3
```

### مرحله ۵: بررسی جای‌نگهدارهای باقی‌مانده

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
```

هر `{{...}}` که موند، دستی پر کن (احتمالاً بخش‌های تحلیلی).

### مرحله ۶: ساخت مقاله

```powershell
python build.py --module a3
```

فایل `paper_full.md` ساخته می‌شود.

---

## 💡 نکات مخصوص A3

### چرا A3 با NHA فرق داره؟

NHA یه attention جدید بود → پارامتر اضافه می‌کرد.
A3 یه **compressor** است → پارامتر **کم** می‌کنه.

پس در `07_results.md`:
- تأکید روی **ratio > 1** (کاهش پارامتر)
- پارامتر `params_after < params_before` باید باشد
- در abstract این رو برجسته کن

### چرا Fisher weighting مهم است؟

بدون Fisher، SVD استاندارد است. با Fisher، فشرده‌سازی روی جهت‌های مهم متمرکز می‌شود.

این رو در `04_method.md` و `08_analysis.md` برجسته کن.

### چرا LayerNorm recalibration؟

بعد از فشرده‌سازی، توزیع activation تغییر می‌کند. LN rescaling این تغییر را جبران می‌کند.

### نکته مهم: `A3FactorizedLinear` پارامتر رو واقعاً کم می‌کنه

```python
# قبل: Linear(768, 768) = 589,824 پارامتر
# بعد: V(64, 768) + U(768, 64) = 98,304 پارامتر
# نسبت: 6× کاهش
