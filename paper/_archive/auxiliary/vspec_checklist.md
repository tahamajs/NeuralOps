# Pre-Submission Checklist — VSpec++

## Setup
- [ ] Copied `_TEMPLATE/` to `vspec/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module vspec`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Three gates stated
- [ ] Variational sampling explained
- [ ] Parameter reduction stated
- [ ] Zero-shot mentioned

## Intro
- [ ] Three orthogonal techniques
- [ ] Problem statement
- [ ] Contributions

## Method
- [ ] SVD factorization
- [ ] Variational spectral gate
- [ ] Head gate
- [ ] Token router
- [ ] Complexity analysis
- [ ] Gate initialization explained

## Results
- [ ] Main accuracy table
- [ ] Attention parameter table
- [ ] Router ablation
- [ ] Spectral gate ablation
- [ ] Head gate ablation
- [ ] Variational vs deterministic
- [ ] Rank ablation
- [ ] LoRA ablation
- [ ] Recovery ablation
- [ ] Gate statistics

## Honesty
- [ ] Attention-only compression stated
- [ ] Cross-attention not supported
- [ ] Variational at inference = mean
- [ ] Single-seed stated
- [ ] Nested encoder routing skipped

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Gate init documented

## Final
- [ ] Read aloud
- [ ] Tables render
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/vspec
nano papers/vspec/variables.json
cd papers/vspec
python apply_vars.py --module vspec
grep -rn "{{" .
python build.py --module vspec --pdf
```
```

---

## 🎯 نکات کلیدی برای این مقاله

### ۱. **مزیت اصلی: سه-Level Gating**

این ماژول **اولین** attention block هست که:
- Token routing
- Variational spectral gating
- Head gating

رو با هم ترکیب می‌کنه. این را در abstract و intro برجسته کن.

### ۲. **Variational — نه فقط Regularization**

Sampling از posterior با reparameterization trick. در training فعال، در inference mean.

### ۳. **Real Parameter Reduction**

SVD فشرده‌سازی **واقعی** روی Q/K/V/O انجام می‌ده. ~۲.۲× reduction روی attention.

### ۴. **Gate Init Bias [3, -3]**

این init به gate می‌گه که در ابتدا near-identity باشه، یعنی SVD representation حفظ بشه.

### ۵. **Nested Encoder Exception**

برای RoBERTa/BERT، token routing **غیرفعال** می‌شه چون context باید حفظ بشه.

### ۶. **تست سریع**

```bash
python run_all_4_vspec.py --models distilbert,roberta,gpt2 --rank 128 --lora_rank 16
```

اگر OOM داد: `--batch_size 4`.

---

**قدم بعدی: `run_all_4_vspec.py` را اجرا کن، اعداد را از `summary.csv` بگیر (شامل gate statistics)، در `variables.json` بگذار.**