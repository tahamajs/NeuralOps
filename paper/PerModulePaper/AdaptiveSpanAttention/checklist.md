# Pre-Submission Checklist — Adaptive Span Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `adaptive_span/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module adaptive_span`
- [ ] No `{{PLACEHOLDER}}` remains

## Content
- [ ] Abstract under 250 words
- [ ] Soft ramp formula stated
- [ ] L1 span penalty explained
- [ ] Learned span results shown

## Method
- [ ] Ramp formula numbered
- [ ] Span logit formula numbered
- [ ] Fast path optimization mentioned
- [ ] Complexity unchanged (stated honestly)

## Results
- [ ] Main table
- [ ] Learned span distribution
- [ ] Per-layer span
- [ ] Max span ablation
- [ ] Ramp size ablation
- [ ] L1 penalty ablation

## Honesty
- [ ] No parameter reduction — stated
- [ ] No speedup without hard truncation — stated
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation

## Interpretability
- [ ] Bimodal span distribution shown
- [ ] Layer-wise pattern shown
- [ ] Comparison with fixed window

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
cp -r papers/_TEMPLATE papers/adaptive_span
nano papers/adaptive_span/variables.json
cd papers/adaptive_span
python apply_vars.py --module adaptive_span
grep -rn "{{" .  # verify empty
python build.py --module adaptive_span --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE adaptive_span
```

### مرحله ۲: اجرای ASA-Span

```powershell
cd D:\SVD_Project
python run_all_4_adaptive_span.py --models distilbert,roberta,gpt2 --max_span 128 --ramp_size 32 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\adaptive_span
python apply_vars.py --module adaptive_span
python build.py --module adaptive_span
