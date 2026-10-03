# Pre-Submission Checklist — MinimalSVD

## Setup
- [ ] Copied `_TEMPLATE/` to `minimal_svd/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module minimal_svd`
- [ ] Verified no `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Real parameter reduction stated
- [ ] Energy-adaptive rank highlighted
- [ ] Four architectures mentioned
- [ ] **Mentioned as minimal baseline**

## Introduction
- [ ] Motivation concrete with numbers
- [ ] Problem statement formal
- [ ] Contributions are 5-6 bullets
- [ ] **Section 1.5 explains relation to NOVA/PRISM**
- [ ] Organization complete

## Method
- [ ] All formulas numbered
- [ ] Algorithm 1 (injection) pseudocode
- [ ] Complexity analysis
- [ ] Energy bound proven

## Architecture
- [ ] DistilBERT (flat) documented
- [ ] RoBERTa (nested) with two-place nested flag
- [ ] GPT-2 (Conv1D) with transposition
- [ ] Qwen3.5 (GQA + RoPE)
- [ ] **Section 5.11 compares with NOVA/PRISM**

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
- [ ] **Table 7.10 (companion comparison)**

## Honesty
- [ ] Real parameter reduction verified
- [ ] Accuracy delta honest
- [ ] No calibration mentioned as feature
- [ ] Single-seed stated
- [ ] **No pre-test / no whole-model traversal stated as limitation**

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HuggingFace
- [ ] Hardware specified
- [ ] Seeds documented
- [ ] Runtime estimates
- [ ] **Whole-model traversal example provided**

## Final Pass
- [ ] Read aloud
- [ ] Numbers match source data
- [ ] All tables render
- [ ] All equations render
- [ ] PDF compiles
- [ ] 2+ external readers

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/minimal_svd
nano papers/minimal_svd/variables.json
cd papers/minimal_svd
python apply_vars.py --module minimal_svd
grep -rn "{{" .  # should be empty
python build.py --module minimal_svd --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE minimal_svd
```

### مرحله ۲: اجرا

```powershell
cd D:\SVD_Project
python run_all_4_minimal_svd.py --models distilbert,roberta,gpt2 --rank 128 --energy 0.95
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_minimal_svd/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply

```powershell
cd D:\SVD_Project\papers\minimal_svd
python apply_vars.py --module minimal_svd
```

### مرحله ۵: بررسی و ساخت

```powershell
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module minimal_svd
```

---

## 💡 نکات کلیدی MinimalSVD

### ۱. **این baseline است، نه variant**

MinimalSVD **پایه‌ای‌ترین** نسخه از خانواده است:
- Injection ساده (single-layer)
- بدون whole-model traversal
- بدون pre-test
- بدون double-injection guard

این را در `01_intro.md` §1.5 و `05_architecture.md` §5.11 برجسته کن.

### ۲. **تفاوت‌ها فقط engineering است**

Table 8.4 نشان می‌دهد که NOVA و PRISM **دقت یکسانی** دارند (اختلاف < 1%). بنابراین:
- Core algorithm یکسان
- فقط injection protocol فرق دارد
- مقاله MinimalSVD باید این را صادقانه بگوید

### ۳. **دو جای nested encoder flag**

MinimalSVD `is_nested_encoder` را در **دو جا** ست می‌کند:
1. در `__init__` بر اساس `model_type`
2. در `copy_weights` بر اساس inspection

**`copy_weights` برنده است.** این را در `B_implementation.md` §B.6 توضیح بده.

### ۴. **Whole-model traversal مسئولیت caller است**

این را در `B_implementation.md` §B.4 نشان بده:

```python
for layer in model.distilbert.transformer.layer:
    inject_minimal_svd(layer, model.config, energy=0.95)
```

### ۵. **فلسفه minimal**

اگر فضای مقاله کم داری، می‌توانی بخش‌های تکراری با NOVA/PRISM را کوتاه کنی چون "the reader is referred to MinimalNOVA for a fuller discussion".

---

**قدم بعدی: `run_all_4_minimal_svd.py` را اجرا کن. اعداد را از `summary.csv` بگیر و در `variables.json` بگذار. اگر خواستی من می‌توانم این کار را برایت انجام دهم — فقط CSV را بفرست.**