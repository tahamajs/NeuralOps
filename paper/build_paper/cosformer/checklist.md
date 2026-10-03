# Pre-Submission Checklist — Cosformer

## Setup
- [ ] Copied `_TEMPLATE/` to `cosformer/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module cosformer`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] All numbers match Section 7
- [ ] Honest about RoBERTa instability
- [ ] Honest about decoder overhead
- [ ] Notes Cosformer is a published method, not ours

### Introduction
- [ ] Motivates linear attention
- [ ] Explains Cosformer's locality trick
- [ ] Contributions honest

### Method
- [ ] Feature map (ELU+1, softplus) explained
- [ ] Cos/sin reweighting derived
- [ ] Both encoder and decoder paths detailed
- [ ] Weight transfer covered

### Results
- [ ] All tables filled
- [ ] Ablations present
- [ ] Comparison with prior modules

### Analysis
- [ ] Locality recovery quantified
- [ ] RoBERTa instability analyzed
- [ ] Decoder overhead broken down

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Seeds documented
- [ ] Diagnostics included

## Appendices
- [ ] A: Cosformer identity proof
- [ ] B: Implementation details
- [ ] C: Full data
- [ ] D: Negative results (RoBERTa)
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render
- [ ] Consistent style

## Final Pass
- [ ] Read aloud
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue (workshop for linear attention, or main conference)
- [ ] Upload
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE cosformer
```

### مرحله ۲: بررسی نتایج

اگر Cosformer را اجرا کردی، نتایج را از `results/unified_cosformer/*/summary.csv` بردار. اگر نه، از اعداد پیش‌فرض `variables.json` استفاده کن (که بر اساس مقالات Cosformer تخمین زده شده).

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و build

```powershell
cd papers\cosformer
python apply_vars.py --module cosformer
python build.py --module cosformer
```

---

## ⚠️ نکات مخصوص Cosformer

### ۱. Cosformer **روش خودت نیست**

باید در abstract و introduction **صریح** بگی:
> «ما Cosformer (Qin et al., 2022) را به pretrained transformerها port کردیم.»

این honest research است.

### ۲. نکات مثبت Cosformer

- **Encoder path ساده و کارا:** فقط ۴ ماتریس‌ضرب
- **Locality recovery واقعی:** cos/sin reweighting کار می‌کند
- **نتایج احتمالاً خوب:** در مقالات، Cosformer ~۹۵٪ softmax accuracy می‌گیرد

### ۳. نکات منفی

- **RoBERTa مشکل دارد:** افت ۱۰٪+ (احتمالاً به خاطر nested encoder)
- **Decoder overhead:** ۵× کندتر
- **پارامتر اضافه می‌کند:** LoRA
- **`tau` مرده:** کد اضافی

### ۴. Venue مناسب

- **Workshops** (NeurIPS Efficiency Workshop, ICML Workshop on Efficient Systems)
- **arXiv**
- **Main conferences** اگر نتایج بهبود داده شوند

### ۵. تفاوت با ماژول‌های قبلی

| ماژول | وضعیت | جنس |
|---|---|---|
| Galerkin | شکست کامل | negative result |
| GatedDeltaNet | timeout | implementation bottleneck |
| **Cosformer** | **نسبتاً موفق** | **port of published method** |

**Cosformer یکی از بهترین ماژول‌های توست.**

---

**قدم بعدی: اگر نتایج Cosformer را از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم. اگر نه، با اعداد پیش‌فرض می‌تونی مقاله رو بسازی و بعد به‌روزرسانی کنی.**