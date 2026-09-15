# Pre-Submission Checklist — Native Sparse Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `nsa/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module nsa`
- [ ] No `{{PLACEHOLDER}}` remains

## Content
- [ ] Abstract under 250 words
- [ ] Local window + top-k described
- [ ] Union mask explained
- [ ] Vectorization emphasized

## Method
- [ ] Local mask formula numbered
- [ ] Top-k formula numbered
- [ ] Causal ordering described
- [ ] Complexity honest

## Results
- [ ] Main table with density column
- [ ] Window size ablation
- [ ] Top-k ablation
- [ ] LoRA rank ablation
- [ ] Density vs L

## Honesty
- [ ] No parameter reduction — stated
- [ ] No time speedup without sparse kernels — stated
- [ ] Failure cases in Appendix D
- [ ] Density vs accuracy trade-off shown

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
cp -r papers/_TEMPLATE papers/nsa
nano papers/nsa/variables.json
cd papers/nsa
python apply_vars.py --module nsa
grep -rn "{{" .  # verify empty
python build.py --module nsa --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE nsa
```

### مرحله ۲: اجرای NSA

```powershell
cd D:\SVD_Project
python run_all_4_nsa.py --models distilbert,roberta,gpt2 --window_size 32 --top_k 16 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\nsa
python apply_vars.py --module nsa
python build.py --module nsa
