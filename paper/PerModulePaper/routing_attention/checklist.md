# Pre-Submission Checklist — Routing Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `routing/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module routing`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Honest about parameter increase
- [ ] Honest about slow inference
- [ ] Notes that real speedups require sparse kernels

### Introduction
- [ ] Motivation concrete
- [ ] Contributions verifiable
- [ ] Paper organization complete

### Method
- [ ] k-means well-documented
- [ ] Cluster-restricted attention clear
- [ ] Token-0 anchor justified
- [ ] Naive complexity honestly stated

### Results
- [ ] All tables filled
- [ ] Parameter count shows increase
- [ ] Time shows increase

## Honesty
- [ ] Not called a compression method
- [ ] Non-differentiable k-means stated
- [ ] Naive loop inefficiency documented
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
Copy-Item -Recurse _TEMPLATE routing
```

### مرحله ۲: اجرای Routing

```powershell
cd D:\SVD_Project
python run_all_4_routing.py --models distilbert,roberta,gpt2 --num_clusters 16 --recovery_steps 50 --lora_rank 16
```

### مرحله ۳: پر کردن `variables.json`

اعداد واقعی از `results/unified_routing/*/summary.csv`.

### مرحله ۴: Apply و build

```powershell
cd papers\routing
python apply_vars.py --module routing
python build.py --module routing
```

---

## 💡 نکات مخصوص Routing

### ۱. مشکل اساسی: کندتر از dense است

در naive implementation، loop روی K cluster یعنی **K بار** محاسبه full attention matrix. برای K=16، این **۱۶ برابر کندتر** است. این رو در Section 8.2 و Abstract واضح بگو.

### ۲. Non-differentiable k-means

`argmin` گرادیان نداره. یعنی cluster assignment یاد نمی‌گیره. این محدودیت بزرگ رو در Section 8.3 برجسته کن.

### ۳. Token-0 anchor یک hack است

اگرچه کار می‌کنه (ablation نشان می‌ده بالای 2% افت می‌ده) ولی توضیح نظری نداره.

### ۴. `RuntimeError` در `copy_weights` یک باگ است

اگر projection صفر باشه، کل injection fail می‌شه. این باید warning باشه نه error.

### ۵. تفاوت با Routing Transformer

Routing Transformer از Q استفاده می‌کنه؛ این از Q+K. تفاوت جزئی ولی قابل ذکر.

### ۶. اگر می‌خواهی سرعت واقعی

باید:
1. از **block-sparse matmul** استفاده کنی
2. یا از **Flash Attention با block mask** استفاده کنی
3. یا **custom Triton kernel** بنویسی

بدون این، Routing فقط یک formulation جالبه، نه یک روش عملی.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از بنچمارک پر کن و build کن. اگر می‌خوای من پر کنم، اعداد `summary.csv` رو بفرست.**