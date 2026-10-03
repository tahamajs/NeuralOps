# Pre-Submission Checklist — ASVD

## Setup
- [ ] Copied `_TEMPLATE/` to `asvd/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module asvd`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Real parameter reduction stated**
- [ ] Accuracy drop stated
- [ ] Energy-based rank mentioned

### Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal
- [ ] Contributions verifiable
- [ ] **Explicit statement this IS a compressor**

### Method
- [ ] All formulas numbered
- [ ] Energy-based rank explained
- [ ] Explicit Σ advantage
- [ ] Complexity analysis
- [ ] Parameter analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows reduction**
- [ ] Rank ablation
- [ ] Energy threshold ablation
- [ ] Dynamic vs fixed rank
- [ ] LoRA rank ablation
- [ ] Error threshold ablation

### Analysis
- [ ] Spectral analysis
- [ ] Per-layer rank distribution
- [ ] Comparison with A3/AASVD
- [ ] `raise` problem discussed

### Limitations
- [ ] Task coverage
- [ ] Cross-attention not supported
- [ ] `raise` on failure
- [ ] Diagonal weighting only
- [ ] No LN recalibration
- [ ] Single-seed

## Honesty
- [ ] Real parameter reduction
- [ ] Failure cases documented
- [ ] Negative results in Appendix D
- [ ] Comparison with published ASVD paper

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
cp -r papers/_TEMPLATE papers/asvd
nano papers/asvd/variables.json
cd papers/asvd
python apply_vars.py --module asvd
grep -rn "{{" .
python build.py --module asvd --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE asvd
```

### مرحله ۲: اجرای ASVD

```powershell
cd D:\SVD_Project
python run_all_4_asvd.py --models distilbert,roberta,gpt2 --rank 64 --energy_threshold 0.95 --error_threshold 0.20 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\asvd
python apply_vars.py --module asvd
python build.py --module asvd
```

---

## 💡 نکات کلیدی ASVD

### ۱. **ASVD یک compressor واقعی است**

برخلاف NHA/EAT/OHA/Infini-Attention:
- پارامتر **کم می‌کند** ✓
- کاهش ۵-۱۲× بسته به rank
- storage و VRAM کاهش می‌یابد

### ۲. **ماتریس Σ جدا**

- `U` (out × rank)
- `Σ` (rank) ← **جدا!**
- `V` (rank × in)

مزیت:
- بازرسی rank
- truncation بدون re-factorization

### ۳. **Energy-based rank**

- `ε = 0.95` → cumulative energy ≥ 95%
- rank تطبیقی per layer
- `r_min = 16`

### ۴. **Pre-flight validation**

`inject_asvd` **قبل** از جایگزینی:
- سعی می‌کند extraction کند
- اگر موفق شد، جایگزین می‌کند
- اگر نه، abort می‌کند (بدون fallback random)

### ۵. **`raise` روی خطا**

- Missing projection → raise
- Reconstruction error > τ → raise

این **بهتر** از silent failure است ولی **سختگیرانه** تر از A3/AASVD.

### ۶. **`set_identity` برای nested encoder**

- RoBERTa: `is_nested_encoder = True`
- `out_proj` را به identity تبدیل می‌کند
- جلوگیری از double projection

### ۷. **مرجع اصلی**

- arXiv:2307.07382
- Yuan et al. (2023)
- این پیاده‌سازی extension است با:
  - Energy-based dynamic rank
  - Explicit Σ
  - Pre-flight validation
  - Multi-architecture

---

**قدم بعدی: `run_all_4_asvd.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**