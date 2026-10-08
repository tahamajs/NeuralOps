# Pre-Submission Checklist — MinimalPRISM

## Setup
- [ ] Copied `_TEMPLATE/` to `minimal_prism/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module minimal_prism`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Real parameter reduction stated
- [ ] Energy-adaptive rank highlighted
- [ ] Four architectures mentioned
- [ ] Injection safety mentioned

## Introduction
- [ ] Motivation concrete with numbers
- [ ] Problem statement formal
- [ ] Contributions are 5-6 bullets
- [ ] Organization complete

## Method
- [ ] All formulas numbered
- [ ] Algorithm 1 (compression) pseudocode
- [ ] Algorithm 2 (injection) pseudocode
- [ ] Complexity analysis
- [ ] Energy bound proven

## Architecture
- [ ] DistilBERT (flat) documented
- [ ] RoBERTa (nested) with injection-time detection
- [ ] GPT-2 (Conv1D) with transposition
- [ ] Qwen3.5 (GQA + RoPE)
- [ ] Failure modes table

## Results
- [ ] Table 7.1 (main) filled
- [ ] Table 7.2 (params) real reduction
- [ ] Table 7.3 (effective rank) per layer
- [ ] Table 7.4 (error) below bound
- [ ] Table 7.5 (efficiency)
- [ ] Table 7.6 (rank cap) ablation
- [ ] Table 7.7 (energy) ablation
- [ ] Table 7.8 (LoRA rank) ablation
- [ ] Table 7.9 (recovery steps) ablation
- [ ] Table 7.10 (comparison with NOVA, A3, AASVD)

## Honesty
- [ ] Real parameter reduction verified
- [ ] Accuracy delta honest
- [ ] No calibration mentioned as feature
- [ ] Single-seed stated
- [ ] No formal downstream bound stated

## Injection Safety
- [ ] Pre-test protocol documented
- [ ] Nested encoder detection at injection time
- [ ] Frobenius validation
- [ ] Double-injection guard

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HuggingFace
- [ ] Hardware specified
- [ ] Seeds documented
- [ ] Runtime estimates

## Final Pass
- [ ] Read aloud
- [ ] Numbers match source data
- [ ] All tables render
- [ ] All equations render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/minimal_prism
nano papers/minimal_prism/variables.json
cd papers/minimal_prism
python apply_vars.py --module minimal_prism
grep -rn "{{" .  # should be empty
python build.py --module minimal_prism --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE minimal_prism
```

### مرحله ۲: اجرای MinimalPRISM

```powershell
cd D:\SVD_Project
python run_all_4_minimal_prism.py --models distilbert,roberta,gpt2 --rank 128 --energy 0.95
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_minimal_prism/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\minimal_prism
python apply_vars.py --module minimal_prism
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module minimal_prism
```

---

## 💡 تفاوت‌های کلیدی PRISM با NOVA

| ویژگی | NOVA | PRISM |
|---|---|---|
| **Injection hook** | پیچیده | **ساده + مقاوم** |
| **Nested detection** | در copy_weights | **در injection-time** |
| **Pre-test** | دارد | **همان (تقویت‌شده)** |
| **Double-injection guard** | دارد | همان |
| **FactorisedLinear** | یکسان | یکسان |
| **Energy-adaptive rank** | یکسان | یکسان |

**نکته برجسته در مقاله:** در `05_architecture.md` بخش `5.7 Injection Algorithm` اشاره کن که PRISM **nested flag را در injection-time ست می‌کند، نه فقط در copy_weights** — این یک بهبود مهندسی است که edge cases را پوشش می‌دهد.

---

## 📌 خلاصه نهایی

MinimalPRISM و MinimalNOVA **از یک پایه مشترک** استفاده می‌کنند:
- `FactorisedLinear` با energy-adaptive rank
- Multi-architecture weight transfer
- LoRA recovery

**تفاوت اصلی در injection safety است** — PRISM ساده‌تر و مقاوم‌تر.

این را در abstract، introduction، و architecture integration برجسته کن.

---

**قدم بعدی: `run_all_4_minimal_prism.py` را اجرا کن. نتایج را از `summary.csv` بگیر و در `variables.json` بگذار. اگر CSV را بفرستی، من می‌توانم `variables.json` را با اعداد واقعی پر کنم.**