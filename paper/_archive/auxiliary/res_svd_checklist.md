# Pre-Submission Checklist — Res-SVD Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `res_svd/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module res_svd`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Two-level SVD stated**
- [ ] **Spectral initialization stated**
- [ ] **Real parameter reduction stated**
- [ ] Equivalence to rank-$(r_b + r_r)$ SVD

### Introduction
- [ ] Motivation concrete
- [ ] **Explicit statement this IS a compressor**
- [ ] PiSSA distinction noted

### Method
- [ ] All formulas numbered
- [ ] Two-level decomposition detailed
- [ ] Spectral init explained
- [ ] Equivalence proof referenced
- [ ] Complexity analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows reduction**
- [ ] Base rank ablation
- [ ] Residual rank ablation
- [ ] Scaling ablation
- [ ] vs pure SVD comparison

### Analysis
- [ ] Why two-level helps
- [ ] Per-layer errors
- [ ] Comparison with PiSSA
- [ ] Silent injection risk

### Limitations
- [ ] **Real compression but PEFT structure**
- [ ] Cross-attention not supported
- [ ] Silent injection failure
- [ ] Equivalence to rank-$(r_b+r_r)$ SVD
- [ ] Single-seed

## Honesty
- [ ] Real parameter reduction
- [ ] Mathematical equivalence documented
- [ ] Silent injection documented
- [ ] Failure cases documented

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Rank hyperparameters documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/res_svd
nano papers/res_svd/variables.json
cd papers/res_svd
python apply_vars.py --module res_svd
grep -rn "{{" .
python build.py --module res_svd --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE res_svd
```

### مرحله ۲: اجرای Res-SVD

```powershell
cd D:\SVD_Project
python run_all_4_res_svd.py --models distilbert,roberta,gpt2 --base_rank 16 --residual_rank 4 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\res_svd
python apply_vars.py --module res_svd
python build.py --module res_svd
```

---

## 💡 نکات کلیدی Res-SVD Attention

### ۱. **Res-SVD یک compressor واقعی است**

- پارامتر **کم** می‌کند: **50\%×**
- برخلاف NHA/EAT/OHA/Infini/SVD-Mamba که پارامتر اضافه می‌کنند

### ۲. **دو سطح SVD**

- **Base**: `U_b V_b^T` (frozen، rank = base_rank)
- **Residual**: `alpha * B A` (trainable، init از spectral residual)

### ۳. **نکته ریاضی کلیدی**

Res-SVD با `(r_b, r_r)` **ریاضیاً معادل** SVD با rank `r_b + r_r` است.

این یعنی Res-SVD **بهتر از SVD خالص نیست** در approximation.
مزیتش در **PEFT structure** است — base فریز می‌شود.

### ۴. **Spectral initialization**

برخلاف LoRA استاندارد که `B = 0` می‌دهد، Res-SVD از 
top-`r_r` singular directions استفاده می‌کند:
- `B = U_r Σ_r^{1/2}`
- `A = Σ_r^{1/2} V_r^T`

### ۵. **تفاوت با PiSSA**

- **PiSSA**: LoRA از principal directions، base از residual
- **Res-SVD**: base از principal directions، LoRA از residual

این دو **دوگان ریاضی** هستند.

### ۶. **مشکلات**

- **Silent injection failure** (`return None`)
- **Cross-attention** not supported
- **Equivalence** به rank-$r_b+r_r$ SVD (مزیت ریاضی ندارد)

### ۷. **مقایسه با ASVD**

| ویژگی | ASVD | Res-SVD |
|---|---|---|
| SVD levels | 1 | **2** |
| Base rank | $r$ | $r_b$ |
| LoRA rank | 0 (zero init) | $r_r$ (spectral init) |
| Params | $r(d_\text{in}+d_\text{out})$ | $(r_b+r_r)(d_\text{in}+d_\text{out})$ |

اگر `r = r_b + r_r`، هر دو **معادل** هستند.
تفاوت در trainable بودن residual.

---

**قدم بعدی: `run_all_4_res_svd.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**