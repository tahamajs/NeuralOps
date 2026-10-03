# Pre-Submission Checklist — ORION

## Setup
- [ ] Copied `_TEMPLATE/` to `orion/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module orion`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Real parameter reduction stated
- [ ] `copy_weights` bug disclosed
- [ ] All three model results reported

### Introduction
- [ ] Three-component approach described
- [ ] Bug disclosure in Section 1.4
- [ ] Contributions honest

### Method
- [ ] SVD factorization clear
- [ ] Head bifurcation explained
- [ ] RFF formula
- [ ] The `copy_weights` bug analyzed

### Results
- [ ] All tables filled
- [ ] Parameter reduction shown
- [ ] Per-layer breakdown
- [ ] Ablations (energy, rank, features, split)

### Analysis
- [ ] `copy_weights` bug analysis
- [ ] `rff_proj` persistence bug
- [ ] Softpick masked row analysis
- [ ] Head split ablation
- [ ] Parameter breakdown

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Seeds documented
- [ ] Diagnostics included

## Appendices
- [ ] A: Math derivations
- [ ] B: Implementation details
- [ ] C: Full data
- [ ] D: Negative results
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render

## Final Pass
- [ ] Read aloud
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue
- [ ] Upload
```

---

## 📄 `build.py` and `apply_vars.py`

Same as previous templates.

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE orion
```

### مرحله ۲: پر کردن `variables.json`

### مرحله ۳: Apply و build

```powershell
cd papers\orion
python apply_vars.py --module orion
python build.py --module orion
```

---

## ⚠️ نکات مخصوص ORION — خلاصه هفت کشف

### ۱. فشرده‌سازی **واقعی**
اولین ماژول تو که پارامتر **کم** می‌کند:
- ۶× کاهش برای 768×768
- ۹.۶× کاهش برای 768×3072

### ۲. باگ `copy_weights`
```python
except Exception as e:
    raise RuntimeError(f"copy_weights failed: {e}")
```
همون باگ AEGIS V4/V5/EAT. **باید fix بشه.**

### ۳. باگ `rff_proj` persistence
```python
persistent = False
```
RFF projection در `state_dict` ذخیره نمی‌شود. **Train/test mismatch** بعد از save/reload.

### ۴. Softpick روی masked rows
```python
p = F.relu(torch.exp(x_f) - 1.0)
return p / (p.sum(dim=dim, keepdim=True) + 1e-8)
```
اگر همه‌ی keys در یک row mask باشند → output = 0.

### ۵. Head split 2/3 : 1/3
```python
self.num_full_heads = max(1, self.num_heads // 3)
self.num_kernel_heads = self.num_heads - self.num_full_heads
```

### ۶. `inject_orion` **safe** است
برخلاف NASC، این یکی `return None` می‌کند:
```python
except Exception as e:
    logger.error("inject_orion failed: %s", e)
    return None
```

### ۷. Python loop در decoder path
فقط ۲ تکرار برای N=128. قابل قبول.

---

### تفاوت با سایر ماژول‌های تو

| ماژول | فشرده‌سازی | باگ کشنده |
|---|---|---|
| FASA | LoRA only | No |
| Cosformer | LoRA only | No |
| NASC | LoRA only | `raise` در inject |
| GoldFinch | +5.7% params | No |
| **ORION** | **-37% params** | **`raise` در copy_weights** |

**ORION یکی از بهترین ماژول‌های توست** — تنها باگش fixable است.

---

## 🎯 توصیه نهایی

اگر می‌خواهی ORION رو در مقاله‌ات به‌عنوان **موفقیت** نشان بدی:
1. باگ `copy_weights` رو fix کن
2. باگ `rff_proj` رو fix کن  
3. Fallback برای softpick اضافه کن
4. دوباره اجرا کن
5. اعداد جدید رو در `variables.json` بگذار

اگر می‌خواهی همون‌طور که هست ثبت کنی:
- **صادقانه** باگ‌ها رو در `08_analysis.md` و `10_limitations.md` بنویس
- تأکید کن روی parameter reduction (~۶×) که **دستاورد اصلی** است

**قدم بعدی: اگر نتایج ORION رو از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم.**