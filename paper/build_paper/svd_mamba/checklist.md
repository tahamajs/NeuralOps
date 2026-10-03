# Pre-Submission Checklist — SVD-Mamba

## Setup
- [ ] Copied `_TEMPLATE/` to `svd_mamba/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module svd_mamba`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Parameter increase stated**
- [ ] Three-path architecture mentioned
- [ ] Gate init strategy mentioned

### Introduction
- [ ] Motivation concrete
- [ ] **Explicit statement this is NOT a compressor**
- [ ] Contributions verifiable

### Method
- [ ] All formulas numbered
- [ ] SSM recurrence explained
- [ ] Chunked scan described
- [ ] Numerical clamps documented
- [ ] Complexity analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows increase**
- [ ] d_state ablation
- [ ] d_conv ablation
- [ ] Gate bias ablation
- [ ] Gate evolution

### Analysis
- [ ] Gate dynamics
- [ ] Numerical stability
- [ ] Parameter breakdown
- [ ] SSM contribution
- [ ] Comparison with pure compressors

### Limitations
- [ ] **Not a compressor**
- [ ] Complex architecture
- [ ] Unstable recovery
- [ ] Cross-attention not supported
- [ ] Single-seed

## Honesty
- [ ] Parameter increase admitted
- [ ] Recovery instability documented
- [ ] Silent injection failure discussed
- [ ] Failure cases documented

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] SSM init documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/svd_mamba
nano papers/svd_mamba/variables.json
cd papers/svd_mamba
python apply_vars.py --module svd_mamba
grep -rn "{{" .
python build.py --module svd_mamba --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE svd_mamba
```

### مرحله ۲: اجرای SVD-Mamba

```powershell
cd D:\SVD_Project
python run_all_4_svd_mamba.py --models distilbert,roberta,gpt2 --d_state 16 --d_conv 4 --global_rank 8 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\svd_mamba
python apply_vars.py --module svd_mamba
python build.py --module svd_mamba
```

---

## 💡 نکات کلیدی SVD-Mamba

### ۱. **SVD-Mamba فشرده‌ساز نیست**

- SVD factorization → projections کم می‌شود
- **ولی** SSM + gating + global → پارامتر **اضافه** می‌کند
- خالص: **+10%** افزایش پارامتر

### ۲. **سه مسیر معماری**

1. Attention (with SVD projections)
2. SSM (Mamba-style selective state space)
3. Global SVD mixing

Fusion via **gate**.

### ۳. **Gate bias = -7**

- `σ(-7) ≈ 0.0009` → SSM فقط 0.1% در ابتدا
- حفظ pretrained behavior
- در طول training تغییر می‌کند (ولی نه به‌شدت)

### ۴. **Chunked scan**

- chunk_size = 64
- Multiple clamps for stability
- Python loop over chunks (کندی)

### ۵. **مشکلات اصلی**

- **افزایش پارامتر** (+10%)
- **کندی** (3.5×)
- **پیچیدگی**
- **Unstable recovery**

### ۶. **مزیت‌ها**

- مسیر SSM برای long-context
- Multi-architecture
- Causal
- نوآوری معماری

### ۷. **مقایسه با AASVD**

| ویژگی | AASVD | SVD-Mamba |
|---|---|---|
| پارامتر | **-50%** | **+10%** |
| Accuracy | -2% | -5% |
| سرعت | کندتر | 3.5× کندتر |
| نوع | Compressor | Attention replacement |

**SVD-Mamba و AASVD کاملاً متفاوت هستند.**

---

**قدم بعدی: `run_all_4_svd_mamba.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**