# Pre-Submission Checklist — Galerkin Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `galerkin/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module galerkin`
- [ ] Verified no `[PLACEHOLDER]` remains

## CRITICAL — Honest Reporting
- [ ] Abstract states negative result for encoders
- [ ] Abstract states GPT-2 is a no-op
- [ ] Section 7.1 shows all three models with honest Δ
- [ ] Section 8 explains why encoders fail
- [ ] Section 8.2 explains why GPT-2 is a no-op
- [ ] Appendix D documents all negative results
- [ ] Conclusion does not overclaim

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] All three model results reported
- [ ] Negative result emphasized
- [ ] No false claims of improvement

### Introduction
- [ ] Motivation clear
- [ ] Contributions honest (includes negative result)
- [ ] Paper organization complete

### Method
- [ ] Galerkin formulation explained
- [ ] DCT basis construction detailed
- [ ] Decoder fallback documented
- [ ] Energy correction discussed

### Results
- [ ] All tables filled
- [ ] Ablations present
- [ ] Comparison with baseline and prior modules

### Analysis
- [ ] Three hypotheses tested
- [ ] H2 (fixed basis) confirmed as primary
- [ ] Attention heatmap analysis
- [ ] Failure taxonomy

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hyperparameters documented
- [ ] Seeds documented

## Appendices
- [ ] A: Mathematical derivations
- [ ] B: Implementation details
- [ ] C: Full data
- [ ] D: Complete negative results
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render
- [ ] Consistent citation style
- [ ] No typos

## Final Pass
- [ ] Read aloud
- [ ] Check every number
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue (consider venues that publish negative results: TMLR, ReScience, etc.)
- [ ] Format per guidelines
- [ ] Upload
- [ ] Announce
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی تمپلیت

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE galerkin
```

### مرحله ۲: بررسی نتایج قبلی

از لاگ‌های قبلی تو:
- DistilBERT: 65.94% (Δ = -24.77%)
- RoBERTa: 48.97% (Δ = -29.59%, collapse)
- GPT-2: 79.36% (Δ = 0.00%, no-op)

### مرحله ۳: پر کردن `variables.json`

مقادیر بالا رو در `variables.json` گذاشتم. اگر داده‌های جدیدتری داری، به‌روز کن.

### مرحله ۴: Apply و build

```powershell
cd papers\galerkin
python apply_vars.py --module galerkin
python build.py --module galerkin
```

---

## ⚠️ نکات مخصوص Galerkin Attention

### ۱. صداقت در مورد negative result

**مهم‌ترین نکته:** این ماژول **شکست خورده**. مقاله باید این رو شفاف بگه. اگه بگی «روی DistilBERT 66% گرفتیم» بدون توضیح اینکه baseline 90.71% بود، گمراه‌کننده‌ست.

### ۲. سه دلیل شکست

1. **Fixed DCT basis نمی‌تونه attention تطبیقی رو یاد بگیره.**
2. **RoBERTa با ۱۲ لایه به chance collapse می‌کنه.**
3. **GPT-2 هیچ تغییری نمی‌کنه چون decoder path fallback به standard attention است.**

### ۳. این ماژول فشرده‌سازی هم نیست

پارامتر **اضافه** می‌کنه (+0.88% DistilBERT، +6.63% RoBERTa). در مقاله این رو واضح بگو.

### ۴. Venue مناسب

مقالات با negative result در venueهای خاصی مثل:
- **TMLR** (Transactions on Machine Learning Research)
- **ReScience C**
- **NeurIPS Datasets and Benchmarks** (نه، این برای datasets)
- arXiv (همیشه)

### ۵. درس اصلی

- **Attention ذاتاً تطبیقی است.** Basis ثابت نمی‌تونه اون رو تقریب بزنه.
- **FNet** با اینکه attention رو کاملاً حذف می‌کنه، بهتر از این ماژول کار می‌کنه. چون DCT در FNet یک **عملیات** است، نه **تقریب**.
- **Linformer** با projection یادگرفتنی بهتر کار می‌کنه.

---

**قدم بعدی: اگر خواستی، می‌تونم `variables.json` رو با اعداد دقیق‌تر از لاگ‌های بنچمارک‌های خودت پر کنم. لاگ‌ها رو بفرست.**