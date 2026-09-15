# Pre-Submission Checklist — Native Hybrid Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `nha/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module nha`
- [ ] No `{{PLACEHOLDER}}` remains

## Content
- [ ] Abstract under 250 words
- [ ] Local + RFF streams described
- [ ] Token-level gate described
- [ ] Orthogonal RFF emphasized

## Method
- [ ] Local stream formula numbered
- [ ] RFF stream formula numbered
- [ ] Chunked scan algorithm complete
- [ ] Complexity is linear in L

## Results
- [ ] Main table
- [ ] Gate distribution
- [ ] Per-layer gate
- [ ] Window size ablation
- [ ] RFF count ablation
- [ ] Gate bias ablation
- [ ] RoBERTa failure reported honestly

## Honesty
- [ ] RoBERTa drop reported (not hidden)
- [ ] No parameter reduction — stated
- [ ] Causality preserved — verified
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
cp -r papers/_TEMPLATE papers/nha
nano papers/nha/variables.json
cd papers/nha
python apply_vars.py --module nha
grep -rn "{{" .  # verify empty
python build.py --module nha --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE nha
```

### مرحله ۲: اجرای NHA

```powershell
cd D:\SVD_Project
python run_all_4_nha.py --models distilbert,roberta,gpt2 --window_size 64 --rff_features 128 --gate_init_bias 3.0 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\nha
python apply_vars.py --module nha
python build.py --module nha
