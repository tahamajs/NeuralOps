# Pre-Submission Checklist — FlashAttention-3 FP8

## Setup
- [ ] Copied `_TEMPLATE/` to `flash3_fp8/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module flash3_fp8`
- [ ] Verified no `[PLACEHOLDER]` remains

## CRITICAL — Hardware Honesty
- [ ] Abstract states FP8 path was NOT taken on RTX 4070
- [ ] Section 6 documents hardware detection
- [ ] Section 7.1 shows detection results
- [ ] Section 8.1 explains why FP8 unavailable
- [ ] Section 10.1 lists this as primary limitation
- [ ] Conclusion mentions no FP8 speedup measured

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Honest about hardware
- [ ] No FP8 speedup claims
- [ ] Accurate numbers

### Introduction
- [ ] Motivation concrete
- [ ] Contributions verifiable
- [ ] Hardware portability emphasized

### Method
- [ ] Detection logic documented
- [ ] Fallback path explained
- [ ] Weight transfer details complete

### Results
- [ ] FP8 path status shown
- [ ] All tables filled
- [ ] No overclaiming

## Reproducibility
- [ ] Hardware check script provided
- [ ] FP8 requirements documented
- [ ] Code on GitHub
- [ ] Checkpoints on HF

## Appendices
- [ ] A: Tiling derivation
- [ ] B: Detection code
- [ ] C: Full data
- [ ] D: Negative results (FP8 unexercised)
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render
- [ ] Consistent style
- [ ] No typos

## Final Pass
- [ ] Read aloud
- [ ] Check every number
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue (arXiv)
- [ ] Format per guidelines
- [ ] Upload
- [ ] Announce
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی تمپلیت

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE flash3_fp8
```

### مرحله ۲: بررسی سخت‌افزار

```powershell
cd D:\SVD_Project
python -c "
import torch
print('GPU:', torch.cuda.get_device_name(0))
print('Capability:', torch.cuda.get_device_capability(0))
print('Float8:', hasattr(torch, 'float8_e4m3fn'))
try:
    import flash_attn
    print('flash_attn:', flash_attn.__version__)
except ImportError:
    print('flash_attn: NOT INSTALLED')
"
```

### مرحله ۳: پر کردن `variables.json`

مقادیر واقعی از خروجی مرحله ۲ و لاگ‌های بنچمارک.

### مرحله ۴: Apply و build

```powershell
cd papers\flash3_fp8
python apply_vars.py --module flash3_fp8
python build.py --module flash3_fp8
```

---

## ⚠️ نکات مخصوص FlashAttention-3 FP8

### ۱. صداقت در مورد FP8

**مهم‌ترین نکته این مقاله:** اگر FP8 path اجرا نشده، **این را در abstract بنویس**. مقاله‌ای که بگه «FP8 روی RTX 4070 تست شد» دروغه.

### ۲. FP8 نیازمندی‌های سخت‌افزاری

| GPU | FP8 Tensor Cores | قابل استفاده در PyTorch؟ |
|---|---|---|
| H100/H200 | ✅ | ✅ با flash-attn-3 |
| RTX 4090 | ✅ | ⚠️ وابسته به driver |
| RTX 4070 | ✅ (Ada) | ❌ معمولاً نه |
| A100 | ❌ | ❌ |
| RTX 3090 | ❌ | ❌ |

### ۳. این ماژول فشرده‌سازی نیست

پارامتر **زیاد** می‌کنه (LoRA). در مقاله این رو واضح بگو.

### ۴. تفاوت با A3 و NHA

- **NHA:** attention جایگزین می‌کنه، پارامتر +0.88%
- **A3:** پارامتر **کم** می‌کنه (5× reduction)
- **flash3_fp8:** پارامتر +0.9%، هدف **سرعت** است نه فشرده‌سازی

### ۵. اگر بعداً H100 گیر آوردی

این تمپلیت آماده‌ست. فقط `FP8_TAKEN` را به `True` تغییر بده، اعداد FP8 را در `07_results.md` پر کن، و بخش‌های مربوط به «FP8 path not taken» را با سرعت‌های واقعی جایگزین کن.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از بنچمارک‌هایت پر کن و `apply_vars.py` را اجرا کن. اگر اعداد را بفرستی، من می‌توانم پر کردن رو انجام بدم.**