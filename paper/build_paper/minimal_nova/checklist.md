# Pre-Submission Checklist — MinimalNOVA

## Setup
- [ ] Copied `_TEMPLATE/` to `minimal_nova/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module minimal_nova`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Real parameter reduction stated
- [ ] Energy-adaptive rank mentioned
- [ ] Multi-architecture support mentioned

## Introduction
- [ ] Motivation concrete
- [ ] Problem statement formal (equation)
- [ ] Contributions verifiable
- [ ] Organization complete

## Method
- [ ] All formulas numbered
- [ ] Algorithm pseudocode complete
- [ ] Complexity analysis
- [ ] Energy bound proven

## Results
- [ ] Main results table (7.1)
- [ ] Parameter analysis (7.2)
- [ ] Rank cap ablation (7.6)
- [ ] Energy ablation (7.7)
- [ ] LoRA rank ablation (7.8)
- [ ] Recovery steps ablation (7.9)
- [ ] Comparison with A3/AASVD (7.10)

## Honesty
- [ ] No calibration data mentioned as a **feature**
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation stated
- [ ] No formal downstream bound stated

## Multi-Architecture
- [ ] DistilBERT (flat) verified
- [ ] RoBERTa (nested) verified
- [ ] GPT-2 (Conv1D) verified
- [ ] Qwen (GQA) verified

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Runtime estimates
- [ ] Seeds documented

## Final Pass
- [ ] Read aloud
- [ ] All tables render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/minimal_nova
nano papers/minimal_nova/variables.json
cd papers/minimal_nova
python apply_vars.py --module minimal_nova
grep -rn "{{" .  # should be empty
python build.py --module minimal_nova --pdf
```
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE minimal_nova
```

### مرحله ۲: اجرا

```powershell
cd D:\SVD_Project
python run_all_4_minimal_nova.py --models distilbert,roberta,gpt2 --rank 128 --energy 0.95
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_minimal_nova/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\minimal_nova
python apply_vars.py --module minimal_nova
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module minimal_nova
```

---

## 💡 نکات کلیدی MinimalNOVA

### ۱. **Energy-Adaptive Rank** — نوآوری اصلی

این تنها چیزی است که MinimalNOVA را از سایر low-rank methods متمایز می‌کند:
```python
cum_energy = cumsum(S²) / sum(S²)
r_eff = min(r_max, first_index_where_energy >= η)
```

در `04_method.md` و `08_analysis.md` این را برجسته کن.

### ۲. **Nested Encoder Bypass** — نکته حیاتی

اگر `is_nested_encoder` را برای RoBERTa اشتباه ست کنی، مدل collapse می‌کند.
در `05_architecture.md` این را با جزئیات توضیح بده.

### ۳. **Conv1D Transposition** — نکته حیاتی

اگر GPT-2 Conv1D را transpose نکنی، ابعاد اشتباه می‌شوند.
در `05_architecture.md` و `B_implementation.md` توضیح بده.

### ۴. **Real Parameter Reduction** — مزیت اصلی

`768×768 = 589,824` → `128×(768+768) = 196,608` → **۳× کاهش**
با `rank=64`: `64×1536 = 98,304` → **۶× کاهش**

این در Table 7.2 برجسته است.

### ۵. **No Calibration** — مزیت منحصر به فرد

برخلاف A3 و AASVD، MinimalNOVA داده calibration نمی‌خواهد.
این را در abstract و discussion برجسته کن.

---

**قدم بعدی: `run_all_4_minimal_nova.py` را اجرا کن. اعداد را از `results/unified_minimal_nova/*/summary.csv` بگیر و در `variables.json` بگذار. اگر خواستی من می‌توانم این کار را برایت انجام دهم — فقط CSV را بفرست.**