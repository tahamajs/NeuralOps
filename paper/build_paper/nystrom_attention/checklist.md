# Pre-Submission Checklist — Nyströmformer

## Setup
- [ ] Copied `_TEMPLATE/` to `nystrom/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module nystrom`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] **Explicitly states: reproduction, not novel method**
- [ ] **Explicitly states: no parameter/KV reduction**
- [ ] Compute reduction emphasized
- [ ] Short sequence fallback mentioned

## Introduction
- [ ] Motivation concrete
- [ ] **Section 1.4: what Nyström does/does not do**
- [ ] Contributions clearly reproduction-focused

## Method
- [ ] Algorithm 1 pseudocode
- [ ] **Three execution paths documented**
- [ ] Landmark pooling described
- [ ] Iterative pinv documented
- [ ] Complexity analysis

## Architecture
- [ ] **Causal decoder limitation stated**
- [ ] **Short sequence fallback stated**
- [ ] Four architectures supported

## Results
- [ ] Table 7.1 (N=128, dense fallback)
- [ ] **Table 7.2 (N=512, Nyström active)**
- [ ] Table 7.4 (landmark ablation)
- [ ] Table 7.5 (pinv iterations)
- [ ] Table 7.7 (companion methods)

## Honesty
- [ ] **Explicitly stated: reproduction study**
- [ ] **Explicitly stated: no parameter reduction**
- [ ] **Explicitly stated: no KV reduction**
- [ ] Short sequence results labeled as "dense fallback"
- [ ] Single-seed stated

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] **Long sequence reproduction documented**
- [ ] Original paper referenced

## Final Pass
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/nystrom
nano papers/nystrom/variables.json
cd papers/nystrom
python apply_vars.py --module nystrom
grep -rn "{{" .  # should be empty
python build.py --module nystrom --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE nystrom
```

### مرحله ۲: اجرای Nyströmformer

```powershell
cd D:\SVD_Project

# Short sequence (N=128, dense fallback)
python run_all_4_nystrom.py --models distilbert,roberta,gpt2 --seq_len 128

# Long sequence (N=512, Nyström active)
python run_all_4_nystrom.py --models distilbert,roberta --seq_len 512 --num_landmarks 32
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_nystrom/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\nystrom
python apply_vars.py --module nystrom
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module nystrom
```

---

## 💡 نکات کلیدی Nyströmformer

### ۱. **این reproduction است، نه نوآوری**

در abstract و introduction **صادقانه** بگو که این کار Xiong et al. (2021) را بازتولید می‌کند.

### ۲. **Compute-only compression**

Nyströmformer **پارامتر کم نمی‌کند** و **KV cache کم نمی‌کند**. فقط **محاسبه attention** را کم می‌کند.

این را در abstract، introduction §1.4، و در همه جداول برجسته کن.

### ۳. **Short sequence fallback**

در $N=128$ (SST-2)، Nyström branch **فعال نمی‌شود**. تمام نتایج اصلی در $N=128$ **dense fallback** هستند.

برای نشان دادن Nyström، باید $N=512$ یا بیشتر تست کنی.

### ۴. **Causal decoder مشکل**

GPT-2 و Qwen از Nyström **نمی‌توانند استفاده کنند** چون causal نیست. همیشه dense causal استفاده می‌کنند.

این را در `05_architecture.md` §5.4 و `10_limitations.md` §10.3 صادقانه بگو.

### ۵. **Iterative pinv vs Direct pinv**

کد تو اول `torch.linalg.pinv` را امتحان می‌کند و اگر شکست خورد به Newton-Schulz می‌رود. این یک **انحراف از مقاله اصلی** است (مقاله فقط Newton-Schulz استفاده می‌کند).

این را در `04_method.md` §4.5 یادداشت کن.

### ۶. **Combining with other methods**

Nyströmformer **orthogonal** است به:
- MinimalSVD (parameter compression)
- SVD-LLM V2 (parameter compression)
- Dobi-SVD (parameter compression)
- MLA (KV compression)

این را در `09_discussion.md` §9.5 برجسته کن.

---

**قدم بعدی: `run_all_4_nystrom.py` را اجرا کن. اعداد را از `summary.csv` بگیر و در `variables.json` بگذار. اگر خواستی من می‌توانم این کار را برایت انجام دهم — فقط CSV را بفرست.**