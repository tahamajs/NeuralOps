# Pre-Submission Checklist — Infini-Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `infini/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module infini`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Parameter increase stated**
- [ ] **"Compressive" clarified** (memory, not model)
- [ ] Accuracy drop stated

### Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal
- [ ] Contributions verifiable
- [ ] **Explicit statement that this is not a compressor**

### Method
- [ ] All formulas numbered
- [ ] β init strategy explained
- [ ] Memory init explained
- [ ] Complexity analysis
- [ ] Parameter analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows increase**
- [ ] β init ablation
- [ ] compress_dim ablation
- [ ] window size ablation
- [ ] LoRA rank ablation

### Analysis
- [ ] β evolution plot
- [ ] Memory utilization
- [ ] Per-layer analysis
- [ ] Failure modes

### Limitations
- [ ] **Not a compressor**
- [ ] Cross-attention not supported
- [ ] Inference overhead at short L
- [ ] Task coverage
- [ ] Single-seed

## Honesty
- [ ] Parameter increase admitted
- [ ] "Compressive" clarified
- [ ] Failure cases documented
- [ ] Negative results in Appendix D

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
cp -r papers/_TEMPLATE papers/infini
nano papers/infini/variables.json
cd papers/infini
python apply_vars.py --module infini
grep -rn "{{" .
python build.py --module infini --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE infini
```

### مرحله ۲: اجرای Infini-Attention

```powershell
cd D:\SVD_Project
python run_all_4_infini.py --models distilbert,roberta,gpt2 --recovery_steps 50 --lora_rank 16 --compress_dim 128 --local_window 128
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\infini
python apply_vars.py --module infini
python build.py --module infini
```

---

## 💡 نکات کلیدی Infini-Attention

### ۱. **"Compressive" به memory اشاره دارد**

نه به compression مدل. این را در abstract، intro و همه جا صادقانه بگو.

### ۲. **پارامتر اضافه می‌کند**

- Memory: `H × d_c × d_h` per layer
- Gate β: 1
- LoRA: 4 projections

برای DistilBERT: **+1.18M پارامتر** (+1.8%)

### ۳. **β = -5 strategy**

- `σ(-5) ≈ 0.0067` → memory فقط 0.7% در ابتدا
- این حفظ pretrained behavior است
- در طول training β یاد می‌گیرد

### ۴. **Local window = 128**

Trade-off:
- کوچک: memory مهم‌تر
- بزرگ: local مهم‌تر

`W = 128` sweet spot است.

### ۵. **مزیت اصلی در long context**

در L=128، Infini کندتر از baseline است. مزیت در L>512 ظاهر می‌شود:
- L=2048: **~10× سریع‌تر**
- L=4096: **~16× سریع‌تر**

### ۶. **Multi-architecture پشتیبانی**

مثل NHA/EAT/AASVD، از flat/nested/causal/GQA پشتیبانی می‌کند.

---

**قدم بعدی: `run_all_4_infini.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**