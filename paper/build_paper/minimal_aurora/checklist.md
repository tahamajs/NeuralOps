# Pre-Submission Checklist — MinimalAURORA

## Setup
- [ ] Copied `_TEMPLATE/` to `minimal_aurora/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module minimal_aurora`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Energy threshold stated**
- [ ] **Real parameter reduction stated**
- [ ] **`pres_gate` issue mentioned**

### Introduction
- [ ] Motivation concrete
- [ ] **Explicit statement this IS a compressor**
- [ ] Contributions verifiable

### Method
- [ ] Energy threshold formula
- [ ] Rank selection detailed
- [ ] Forward pass
- [ ] Complexity analysis
- [ ] **`pres_gate` documented**

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows reduction**
- [ ] Energy threshold ablation
- [ ] Max rank cap ablation
- [ ] LoRA rank ablation
- [ ] `output_scale` tracking

### Analysis
- [ ] Why energy threshold works
- [ ] Per-layer rank distribution
- [ ] **`pres_gate` dead code analysis**
- [ ] `output_scale` drift

### Limitations
- [ ] No activation weighting
- [ ] No LayerNorm recal
- [ ] **`pres_gate` unused**
- [ ] **`output_scale` drift**
- [ ] Silent injection failure
- [ ] Single-seed

## Honesty
- [ ] Real parameter reduction
- [ ] **`pres_gate` dead code documented**
- [ ] `output_scale` drift documented
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Energy thresholds documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/minimal_aurora
nano papers/minimal_aurora/variables.json
cd papers/minimal_aurora
python apply_vars.py --module minimal_aurora
grep -rn "{{" .
python build.py --module minimal_aurora --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE minimal_aurora
```

### مرحله ۲: اجرای MinimalAURORA

```powershell
cd D:\SVD_Project
python run_all_4_minimal_aurora.py --models distilbert,roberta,gpt2 --energy 0.95 --rank 64 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\minimal_aurora
python apply_vars.py --module minimal_aurora
python build.py --module minimal_aurora
```

---

## 💡 نکات کلیدی MinimalAURORA

### ۱. **MinimalAURORA یک compressor واقعی است**

- SVD با energy threshold → پارامتر **کم** می‌کند
- نسبت کاهش: **~50\%×**
- برخلاف NHA/EAT/OHA/Infini/SVD-Mamba

### ۲. **Energy-based rank selection**

- `E(r) ≥ ε` با `ε = 0.95` پیش‌فرض
- هر لایه rank متفاوت
- ساده‌تر از AASVD (نیاز به calibration ندارد)

### ۳. **مشکلات کد**

- **`pres_gate` استفاده نمی‌شود** — dead code، توصیه به حذف
- **`output_scale`** ممکن است drift کند (ولی نه به 0)
- **Silent injection failure** (`return None`)

### ۴. **تفاوت با AASVD/A3**

| ویژگی | AASVD | A3 | MinimalAURORA |
|---|---|---|---|
| Weighting | Full cov | Fisher diag | **None** |
| Calibration | Required | Required | **No** |
| Rank | Fixed cap | Adaptive retry | **Energy** |
| Complexity | High | Medium | **Low** |

### ۵. **مقایسه با Res-SVD**

| ویژگی | Res-SVD | MinimalAURORA |
|---|---|---|
| SVD levels | 2 | **1** |
| LoRA init | Spectral | **Zero** |
| `pres_gate` | No | **Yes (unused)** |

### ۶. **مزیت اصلی**

- **سادگی**: هیچ calibration، هیچ weighting
- **Multi-architecture**: 4 خانواده
- **Speed**: کامپایلش سریع‌تر از AASVD

### ۷. **معایب اصلی**

- **`pres_gate` dead code** — نشانه‌ای از code quality ضعیف
- **Silent failure** در injection
- **بدون activation weighting** → دقت کمتر از A3/AASVD

### ۸. **Recommendation**

قبل از انتشار مقاله:
1. `pres_gate` را حذف کن
2. `output_scale` را فریز کن یا حذف
3. sanity check بعد از injection اضافه کن
4. benchmark با A3/AASVD/ASVD برای مقایسه

---

**قدم بعدی: `run_all_4_minimal_aurora.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**