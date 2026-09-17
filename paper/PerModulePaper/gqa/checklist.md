# Pre-Submission Checklist — GQA

## Setup
- [ ] Copied `_TEMPLATE/` to `gqa/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module gqa`
- [ ] No `[PLACEHOLDER]` remains

## Content
- [ ] Abstract under 250 words
- [ ] KV reduction stated clearly
- [ ] Accuracy delta honest
- [ ] $H_{KV}$ value specified
- [ ] All four architectures documented

## Method
- [ ] All formulas numbered
- [ ] Algorithm pseudocode complete
- [ ] KV cache analysis detailed
- [ ] Complexity table filled

## Results
- [ ] Main table filled
- [ ] KV heads ablation
- [ ] LoRA rank ablation
- [ ] Recovery steps ablation
- [ ] Comparison with companions

## Honesty
- [ ] GQA reduces KV cache, not parameters — stated explicitly
- [ ] Parameter reduction is modest — not overclaimed
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hardware specified
- [ ] Seeds documented

## Final
- [ ] Read aloud
- [ ] PDF compiles
- [ ] External feedback

---

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/gqa
nano papers/gqa/variables.json
cd papers/gqa
python apply_vars.py --module gqa
grep -rn "{{" .  # verify empty
python build.py --module gqa --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE gqa
```

### مرحله ۲: اجرای GQA

```powershell
cd D:\SVD_Project
python run_all_4_gqa.py --models distilbert,roberta,gpt2 --num_kv_heads 6 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_gqa/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\gqa
python apply_vars.py --module gqa
python build.py --module gqa
