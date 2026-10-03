# Pre-Submission Checklist — SALT

## Setup
- [ ] Copied `_TEMPLATE/` to `salt/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module salt`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Storage reduction claimed correctly
- [ ] Inference memory limitation stated
- [ ] No overclaiming

### Introduction
- [ ] Motivation concrete
- [ ] Contributions verifiable
- [ ] Paper organization complete

### Method
- [ ] SVD buffers explained
- [ ] γ modulation explained
- [ ] AB residual explained
- [ ] Naive vs fused forward clearly distinguished

### Results
- [ ] All tables filled
- [ ] Parameter reduction verified
- [ ] Storage reduction verified
- [ ] Inference time honestly reported

## Honesty
- [ ] Not claimed as inference accelerator
- [ ] Fused forward missing documented
- [ ] Bugs documented in §10
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Appendices
- [ ] A: Math derivations
- [ ] B: Implementation
- [ ] C: Full data
- [ ] D: Negative results
- [ ] E: Ethics

## Final Pass
- [ ] Read aloud
- [ ] Check every number
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue
- [ ] Format
- [ ] Upload
- [ ] Announce
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی تمپلیت

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE salt
```

### مرحله ۲: اجرای SALT

```powershell
cd D:\SVD_Project
python run_all_4_salt.py --models distilbert,roberta,gpt2 --rank 16 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و build

```powershell
cd papers\salt
python apply_vars.py --module salt
python build.py --module salt
```

---

## 💡 نکات مخصوص SALT

### ۱. SALT **واقعاً** پارامتر کم می‌کند (برخلاف NHA/RALA/routing)

این تنها مزیت اصلی SALT است. در Abstract و Introduction برجسته کن.

### ۲. اما **inference memory کم نمی‌شود**

چون forward full W را reconstruct می‌کند. این بزرگ‌ترین limitation است.

### ۳. Fused forward راه‌حل است

```python
z = F.linear(x, self.Vt)
z = z * (self.S * self.gamma)
y_svd = F.linear(z, self.U)
y_lora = F.linear(F.linear(x, self.B), self.A)
```

این را در Section 10 به عنوان "fix" پیشنهاد بده.

### ۴. سه باگ اصلی:

1. `raise RuntimeError` در `copy_weights` → باید warning باشه
2. `raise RuntimeError` روی Frobenius small → باید warning باشه
3. `output_scale` dead code → باید حذف بشه

### ۵. تفاوت با A3

- **A3:** activation-aware SVD با LayerNorm recalibration → واقعاً compress می‌کند
- **SALT:** SVD + trainable γ + LoRA → storage compression با adaptation

### ۶. تفاوت با LoRA

- **LoRA:** base W کامل ذخیره می‌شه + A, B کوچک
- **SALT:** فقط top-r از W + A, B

### ۷. تفاوت با PiSSA

- **PiSSA:** full W ذخیره می‌شه، LoRA از SVD init می‌شه
- **SALT:** فقط truncated SVD ذخیره می‌شه + γ

---

**قدم بعدی: `variables.json` را با اعداد واقعی از بنچمارک پر کن. اگر summary.csv را بفرستی، من پر می‌کنم.**