# Pre-Submission Checklist — Adaptive SVD

## Setup
- [ ] Copied `_TEMPLATE/` to `adaptive_svd/`
- [ ] Filled `variables.json` with real values
- [ ] Ran `python apply_vars.py --module adaptive_svd`
- [ ] No `{{PLACEHOLDER}}` remains: `Select-String -Path "*.md" -Pattern "\{\{"`

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Calibration-free nature highlighted
- [ ] Energy threshold τ stated
- [ ] Real parameter reduction stated
- [ ] Per-layer rank range stated

### Introduction
- [ ] Motivation concrete (not "transformers are popular")
- [ ] Problem statement formal
- [ ] Contributions verifiable
- [ ] Paper organization complete

### Related Work
- [ ] Covers classical SVD and rank selection
- [ ] Covers activation-aware methods (A3, AASVD)
- [ ] Covers PEFT (LoRA, AdaLoRA, PiSSA)
- [ ] Includes "position of our work" section
- [ ] All citations are real

### Method
- [ ] Every formula numbered
- [ ] Every formula referenced in text
- [ ] Algorithm 1 and 2 written
- [ ] Complexity analysis complete
- [ ] T5/GQA/RoPE support described

### Results
- [ ] Table 7.1 main results with all 3 models
- [ ] Table 7.3 per-layer rank distribution
- [ ] Table 7.6 energy threshold ablation
- [ ] Table 7.7 residual rank ablation
- [ ] Table 7.8 recovery steps ablation
- [ ] Table 7.9 rank cap ablation
- [ ] Table 7.10 companion comparison
- [ ] All ablations have observations

### Analysis
- [ ] Explains why it works
- [ ] Per-layer rank distribution analyzed
- [ ] Comparison with A3/AASVD
- [ ] Comparison with fixed rank
- [ ] Numerical stability discussed

### Limitations
- [ ] Task coverage stated
- [ ] Model coverage stated
- [ ] Hardware coverage stated
- [ ] Inference overhead stated
- [ ] Single-seed limitation stated
- [ ] T5/Qwen verification status stated

## Reproducibility

- [ ] Code on GitHub
- [ ] Checkpoints on HuggingFace
- [ ] Hardware specified
- [ ] Seeds documented
- [ ] Dependencies pinned
- [ ] Runtime estimates provided
- [ ] Reproduction commands tested
- [ ] Per-layer ranks documented

## Honesty

- [ ] No overclaiming
- [ ] Failure cases in Appendix D
- [ ] Negative results documented
- [ ] Single-seed transparency
- [ ] "Calibration-free" accurate (not calibration-ignoring)
- [ ] Parameter reduction verified

## Formatting

- [ ] All tables render
- [ ] All equations render
- [ ] Consistent citation style
- [ ] No broken links
- [ ] All `{{PLACEHOLDER}}` substituted

## Ethics

- [ ] Broader impact discussed
- [ ] Environmental cost estimated
- [ ] Bias concerns stated
- [ ] Misuse risks stated
- [ ] Funding disclosed
- [ ] COI disclosed
- [ ] LLM usage disclosed

## Final Pass

- [ ] Read entire paper aloud
- [ ] Verify all numbers against `summary.csv`
- [ ] Cross-check tables with data
- [ ] Get feedback from 2+ readers
- [ ] Address all feedback
- [ ] PDF compiles
- [ ] Ready to submit

## Quick Start

```bash
# 1. Copy template
cp -r papers/_TEMPLATE papers/adaptive_svd

# 2. Fill variables.json
nano papers/adaptive_svd/variables.json

# 3. Apply
cd papers/adaptive_svd
python apply_vars.py --module adaptive_svd

# 4. Verify
grep -rn "{{" .
# Should be empty

# 5. Build
python build.py --module adaptive_svd --pdf

# 6. Review and submit
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی تمپلیت

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE adaptive_svd
```

### مرحله ۲: اجرای Adaptive SVD

```powershell
cd D:\SVD_Project
python run_all_4_adaptive_svd.py --models distilbert,roberta,gpt2 --energy_threshold 0.92 --residual_rank 16 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از فایل‌های `results/unified_adaptive_svd/*/summary.csv` و `metrics.json` 
اعداد واقعی را بردار. `metrics.json` شامل `layer_ranks` است که per-layer 
rank را می‌دهد.

### مرحله ۴: Apply و بررسی

```powershell
cd D:\SVD_Project\papers\adaptive_svd
python apply_vars.py --module adaptive_svd
Select-String -Path "*.md" -Pattern "\{\{"
```

### مرحله ۵: ساخت مقاله

```powershell
python build.py --module adaptive_svd
# یا با PDF:
python build.py --module adaptive_svd --pdf
