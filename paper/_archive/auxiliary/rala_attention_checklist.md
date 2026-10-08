# Pre-Submission Checklist — RALA

## Setup
- [ ] Copied `_TEMPLATE/` to `rala/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module rala`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Honest about parameter increase
- [ ] Honest about slow inference
- [ ] No overclaiming

### Introduction
- [ ] Motivation concrete
- [ ] Contributions verifiable
- [ ] Paper organization complete

### Method
- [ ] Softplus kernel defined
- [ ] Rank augmentation defined
- [ ] Scalar gate described
- [ ] Sequential scan documented

### Results
- [ ] All tables filled
- [ ] Parameter count shows increase
- [ ] Time shows increase

## Honesty
- [ ] Not called a compression method
- [ ] Frozen U, V documented
- [ ] Sequential scan documented
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Appendices
- [ ] A: Math derivations
- [ ] B: Implementation code
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
Copy-Item -Recurse _TEMPLATE rala
```

### مرحله ۲: اجرای RALA

```powershell
cd D:\SVD_Project
python run_all_4_rala.py --models distilbert,roberta,gpt2 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json`

اعداد واقعی از `results/unified_rala/*/summary.csv`.

### مرحله ۴: Apply و build

```powershell
cd papers\rala
python apply_vars.py --module rala
python build.py --module rala
```

---

## 💡 نکات مخصوص RALA

### ۱. RALA یک attention جایگزین است، نه compressor

پارامتر **اضافه** می‌کنه (U, V, alpha + LoRA). در مقاله این رو واضح بگو.

### ۲. مشکل Frozen U, V

U و V با `randn * 0.01` راه‌اندازی می‌شن و بعد **freeze** می‌شن. فقط `alpha` trainable است. یعنی augmentation یک projection رندوم ثابت ضربدر یک اسکالر یادگیرنده است. این محدودیت رو در Section 10 برجسته کن.

### ۳. Causal scan فوق‌العاده کند است

Python loop روی sequence length. برای L=128، این 128 × 4 = 512 kernel launch در هر لایه. در Section 8.2 توضیح بده.

### ۴. سه نوآوری RALA

1. **Softplus kernel:** هموارتر از elu+1
2. **Rank-r augmentation:** per-head matrix
3. **Scalar gate:** ترکیب خطی

### ۵. تفاوت با Delta Net

Delta Net از update rule داده‌محور استفاده می‌کنه؛ RALA از یک scalar gate. RALA ساده‌تره ولی expressive کمتر.

---

**قدم بعدی: `variables.json` را با اعداد واقعی پر کن و build کن. اگر خواستی، اعداد را بفرست تا پر کردن انجام بشه.**