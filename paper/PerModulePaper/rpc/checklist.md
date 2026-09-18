# Pre-Submission Checklist — RPC

## Setup
- [ ] Copied `_TEMPLATE/` to `rpc/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module rpc`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content

### Abstract
- [ ] Under 250 words
- [ ] **Parameter increase stated**
- [ ] **No memory reduction stated**
- [ ] JL lemma mentioned
- [ ] Accuracy drop stated

### Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal
- [ ] **Explicit statement this IS NOT a compressor**
- [ ] **Explicit statement NO memory reduction**

### Method
- [ ] All formulas numbered
- [ ] JL connection clear
- [ ] Sketch mechanism explained
- [ ] Complexity analysis
- [ ] Parameter analysis

### Results
- [ ] Main table complete
- [ ] **Parameter analysis shows increase**
- [ ] **VRAM analysis shows no change**
- [ ] Sketch rank ablation
- [ ] LoRA rank ablation
- [ ] Actual vs theoretical speedup

### Analysis
- [ ] Why RPC has modest gains
- [ ] When RPC helps (L > 512)
- [ ] Memory problem explicit
- [ ] Sketch quality
- [ ] `raise` problem

### Limitations
- [ ] **Not a compressor**
- [ ] **No memory reduction**
- [ ] Only helps at long sequences
- [ ] Cross-attention not supported
- [ ] Single-seed

## Honesty
- [ ] Parameter increase admitted
- [ ] No memory reduction admitted
- [ ] Failure cases documented
- [ ] Theoretical vs actual speedup gap discussed

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Random seed for `proj` documented

## Final Pass
- [ ] Read aloud
- [ ] All numbers verified
- [ ] Tables render
- [ ] PDF compiles

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/rpc
nano papers/rpc/variables.json
cd papers/rpc
python apply_vars.py --module rpc
grep -rn "{{" .
python build.py --module rpc --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE rpc
```

### مرحله ۲: اجرای RPC

```powershell
cd D:\SVD_Project
python run_all_4_rpc.py --models distilbert,roberta,gpt2 --rank 32 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json` از `summary.csv`

### مرحله ۴: Apply و Build

```powershell
cd D:\SVD_Project\papers\rpc
python apply_vars.py --module rpc
python build.py --module rpc
```

---

## 💡 نکات کلیدی RPC

### ۱. **RPC فشرده‌ساز نیست**

- پارامتر **اضافه** می‌کند (LoRA)
- حافظه **کم نمی‌کند** (score matrix $L \times L$)
- فقط FLOPs attention score را کم می‌کند

### ۲. **"Compressed" در اسم به sketch اشاره دارد**

نه به compression مدل. این را همه جا صادقانه بگو.

### ۳. **JL lemma مبنای ریاضی**

Johnson-Lindenstrauss (1984): random projection distances را حفظ می‌کند.
- `r ≥ O(log N / ε²)`
- برای L=128، r=32 کافی است

### ۴. **مشکل اساسی: memory**

Score matrix **هنوز L × L** است. FlashAttention-style memory saving 
**حاصل نمی‌شود**.

### ۵. **مزیت واقعی در L > 512**

در L=128، RPC **کندتر** از baseline است.
در L=1024، **~1.7× سریع‌تر**.

### ۶. **تفاوت با کمپانیون‌ها**

| ماژول | چه چیزی را کم می‌کند |
|---|---|
| AASVD/ASVD | پارامتر (و memory) |
| EAT | token count |
| OHA | sequence length |
| **RPC** | **attention FLOPs** |

### ۷. **`raise` روی خطا**

RPC سریع می‌میرد اگر projection گم شود. مثل ASVD/OHA.

---

**قدم بعدی: `run_all_4_rpc.py` را اجرا کن و `summary.csv` را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**