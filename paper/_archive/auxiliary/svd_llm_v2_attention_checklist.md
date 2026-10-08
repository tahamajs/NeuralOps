# Pre-Submission Checklist — SVD-LLM V2

## Setup
- [ ] Copied `_TEMPLATE/` to `svd_llm_v2/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module svd_llm_v2`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Explicit U/Σ/V^H storage highlighted**
- [ ] Real parameter reduction stated
- [ ] Energy threshold mentioned
- [ ] GQA pooling mentioned
- [ ] Pre-test mentioned

## Introduction
- [ ] Motivation concrete with numbers
- [ ] Problem statement formal
- [ ] **Section 1.5 explains relation to prior SVD-LLM version**
- [ ] Contributions are 5-6 bullets

## Method
- [ ] All formulas numbered
- [ ] Explicit factorization formula clear
- [ ] `_project` helper documented
- [ ] Energy bound proven

## Architecture
- [ ] **GQA mean pooling detailed**
- [ ] Conv1D handling
- [ ] Nested encoder detection
- [ ] Pre-test protocol

## Results
- [ ] Table 7.1 (main) filled
- [ ] Table 7.2 (params)
- [ ] Table 7.3 (effective rank)
- [ ] **Table 7.4 (singular values)** — unique to this paper
- [ ] Table 7.5 (error)
- [ ] Table 7.6-7.9 (ablations)
- [ ] **Table 7.10 (explicit vs symmetric)** — unique to this paper
- [ ] Table 7.11 (companion comparison)

## Honesty
- [ ] Real parameter reduction verified
- [ ] Accuracy delta honest
- [ ] **FP16 sensitivity stated** (unique limitation)
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Explicit storage documented

## Final Pass
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/svd_llm_v2
nano papers/svd_llm_v2/variables.json
cd papers/svd_llm_v2
python apply_vars.py --module svd_llm_v2
grep -rn "{{" .  # should be empty
python build.py --module svd_llm_v2 --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE svd_llm_v2
```

### مرحله ۲: اجرای SVD-LLM V2

```powershell
cd D:\SVD_Project
python run_all_4_svd_llm_v2.py --models distilbert,roberta,gpt2 --rank 128 --energy 0.95
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_svd_llm_v2/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\svd_llm_v2
python apply_vars.py --module svd_llm_v2
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module svd_llm_v2
```

---

## 💡 نکات مخصوص SVD-LLM V2

### ۱. **Explicit factorization** — مزیت اصلی

این تنها روشی است که $U$, $\Sigma$, $V^H$ را جدا ذخیره می‌کند. در abstract و introduction برجسته کن.

### ۲. **Singular value statistics** — بخش یکتای این مقاله

Table 7.4 و Section 8.2 **فقط** در این مقاله معنی دارند چون فقط این روش $\Sigma$ را حفظ می‌کند.

### ۳. **FP16 sensitivity** — limitation یکتا

این تنها روشی است که ممکن است در FP16 مشکل داشته باشد. صادقانه در Section 10.4 توضیح بده.

### ۴. **GQA mean pooling** — جزئیات مهم

`copy_weights` این کار را می‌کند:
```python
rep = w_k.shape[0] // self.kv_head_size
w_k_pooled = w_k.view(self.num_kv_heads, rep, self.head_dim, -1).mean(dim=1)
```

این را در Section 5.5 و B.4 توضیح بده.

### ۵. **Separation from companion methods**

SVD-LLM V2 **نباید** به‌عنوان "نسخه‌ای از MinimalSVD" معرفی شود. این یک پیاده‌سازی **جداگانه** با انتخاب طراحی متفاوت است:
- Symmetric vs Explicit
- Same accuracy, different flexibility

---

**قدم بعدی: `run_all_4_svd_llm_v2.py` را اجرا کن. اعداد را از `summary.csv` بگیر و در `variables.json` بگذار. اگر خواستی من می‌توانم این کار را برایت انجام دهم — فقط CSV را بفرست.**