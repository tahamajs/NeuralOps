# Pre-Submission Checklist — Primal Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `primal/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module primal`
- [ ] No `{{PLACEHOLDER}}` remains

## Content
- [ ] Abstract under 250 words
- [ ] Asymmetry clearly stated as the key innovation
- [ ] Schulz iteration explained
- [ ] Attention memory reduction stated

## Method
- [ ] All formulas numbered
- [ ] Complexity analysis correct (linear in L)
- [ ] Associativity shown explicitly
- [ ] Landmark selection detailed

## Results
- [ ] Main table with attn memory column
- [ ] $m$ ablation
- [ ] $K$ ablation
- [ ] Scaling with $L$

## Honesty
- [ ] No parameter reduction — stated explicitly
- [ ] Causality approximation — noted
- [ ] Short-L slowdown — acknowledged
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
cp -r papers/_TEMPLATE papers/primal
nano papers/primal/variables.json
cd papers/primal
python apply_vars.py --module primal
grep -rn "{{" .  # verify empty
python build.py --module primal --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE primal
```

### مرحله ۲: اجرای Primal

```powershell
cd D:\SVD_Project
python run_all_4_primal.py --models distilbert,roberta,gpt2 --rank 64 --pinv_iters 4 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

از `results/unified_primal/*/summary.csv` اعداد واقعی را بردار.

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\primal
python apply_vars.py --module primal
python build.py --module primal
