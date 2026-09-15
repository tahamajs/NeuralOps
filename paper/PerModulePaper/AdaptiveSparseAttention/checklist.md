# Pre-Submission Checklist — Adaptive Sparse Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `adaptive_sparse/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module adaptive_sparse`
- [ ] No `{{PLACEHOLDER}}` remains

## Content
- [ ] Abstract under 250 words
- [ ] Learned per-head sparsity is clear
- [ ] Soft top-k explained
- [ ] Chunked memory saving stated

## Method
- [ ] Sparsity logits formula numbered
- [ ] Soft mask formula numbered
- [ ] Chunked complexity table complete
- [ ] Gradient flow explained

## Results
- [ ] Main table
- [ ] Sparsity ablation
- [ ] Temperature ablation
- [ ] Head budget distribution
- [ ] Attention mass retention

## Honesty
- [ ] No parameter reduction — stated
- [ ] No time reduction — stated
- [ ] Causality approximation — noted
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
cp -r papers/_TEMPLATE papers/adaptive_sparse
nano papers/adaptive_sparse/variables.json
cd papers/adaptive_sparse
python apply_vars.py --module adaptive_sparse
grep -rn "{{" .  # verify empty
python build.py --module adaptive_sparse --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE adaptive_sparse
```

### مرحله ۲: اجرای ASA

```powershell
cd D:\SVD_Project
python run_all_4_adaptive_sparse.py --models distilbert,roberta,gpt2 --sparsity_ratio 0.5 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\adaptive_sparse
python apply_vars.py --module adaptive_sparse
python build.py --module adaptive_sparse
