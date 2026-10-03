# Pre-Submission Checklist — UnifiedRoPE

## Setup
- [ ] Copied `_TEMPLATE/` to `unified_rope/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module unified_rope`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Explicitly states: replacement, not compression**
- [ ] **Explicitly states: no parameter reduction**
- [ ] Four architectures mentioned
- [ ] RoPE emphasized

## Introduction
- [ ] Motivation concrete
- [ ] **Section 1.4: is NOT a compression method**
- [ ] Contributions reproduction-focused

## Method
- [ ] Algorithm described
- [ ] **Two RoPE entry points documented**
- [ ] **Precomputed cache documented**
- [ ] GQA expansion documented

## Architecture
- [ ] **Conv1D transposition documented**
- [ ] **Nested encoder bypass documented**
- [ ] **GQA expansion documented**
- [ ] Four families supported

## Results
- [ ] Table 7.1 (main) filled
- [ ] **Table 7.2 (params 1.00×)** — emphasizes no compression
- [ ] **Table 7.3 (norm preservation)**
- [ ] **Table 7.4 (relative position)**
- [ ] Table 7.5 (efficiency)
- [ ] Tables 7.6-7.9 (ablations)

## Honesty
- [ ] **Explicitly stated: replacement, not compression**
- [ ] **Explicitly stated: no param/KV/compute reduction**
- [ ] **Added RoPE may reduce accuracy**
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] RoPE correctness verification included

## Final Pass
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/unified_rope
nano papers/unified_rope/variables.json
cd papers/unified_rope
python apply_vars.py --module unified_rope
grep -rn "{{" .  # should be empty
python build.py --module unified_rope --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE unified_rope
```

### مرحله ۲: اجرای UnifiedRoPE

```powershell
cd D:\SVD_Project
python run_all_4_unified_rope.py --models distilbert,roberta,gpt2 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_unified_rope/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\unified_rope
python apply_vars.py --module unified_rope
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module unified_rope
```

---

## 💡 نکات کلیدی UnifiedRoPE

### ۱. **این compression نیست** — مهم‌ترین نکته

در abstract، introduction §1.4، و همه جداول **صادقانه** بگو که UnifiedRoPE **هیچ پارامتر/KV/compute کاهش نمی‌دهد**.

Table 7.2 باید `1.00×` برای همه مدل‌ها نشان بدهد.

### ۲. **Added RoPE مشکل‌ساز است**

DistilBERT، RoBERTa، و GPT-2 با **absolute positions** آموزش دیده‌اند. اضافه کردن RoPE باعث تغییر الگوی attention می‌شود.

این را در abstract و introduction §1.3 صادقانه بگو.

### ۳. **سه چالش engineering**

- **Conv1D transposition** برای GPT-2
- **Nested encoder bypass** برای RoBERTa
- **GQA expansion** برای Qwen

هر کدام یک subsection در `05_architecture.md` داشته باشند.

### ۴. **RoPE Correctness verification**

Table 7.3 (norm preservation) و Table 7.4 (relative position) **یکتا** به این مقاله هستند. از آن‌ها استفاده کن.

### ۵. **Orthogonal to compression**

در `09_discussion.md` §9.5 جدول بگذار که نشان بده UnifiedRoPE **orthogonal** است به:
- MinimalSVD (params)
- MLA (KV)
- Nyströmformer (compute)

### ۶. **Non-persistent buffers**

cache غیر-persistent است. این را در `04_method.md` §4.2.2 و `10_limitations.md` §10.10 توضیح بده.

### ۷. **Safe injection raises**

`inject_unified_rope` در failure **raise** می‌کند (برخلاف بعضی modules که `None` برمی‌گردانند). این را در `10_limitations.md` §10.10 بگو.

---

**قدم بعدی: `run_all_4_unified_rope.py` را اجرا کن. اعداد را از `summary.csv` بگیر و در `variables.json` بگذار. اگر CSV را بفرستی، من می‌توانم `variables.json` را با اعداد واقعی پر کنم.**