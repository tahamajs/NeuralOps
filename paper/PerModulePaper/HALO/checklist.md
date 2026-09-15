# Pre-Submission Checklist — HALO-ReGLA

## Setup
- [ ] Copied `_TEMPLATE/` to `halo_regla/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module halo_regla`
- [ ] No `{{PLACEHOLDER}}` remains

## Content
- [ ] Abstract under 250 words
- [ ] Dense + linear streams described
- [ ] Per-head gate described
- [ ] Gate initialization explained

## Method
- [ ] Dense stream formula numbered
- [ ] Linear stream formula numbered
- [ ] Gate formula numbered
- [ ] Prefix-sum causality proven

## Results
- [ ] Main table
- [ ] Gate distribution
- [ ] Per-layer gate
- [ ] Gate bias ablation
- [ ] Kernel ablation
- [ ] LoRA rank ablation

## Honesty
- [ ] No parameter reduction — stated
- [ ] No speedup without hard routing — stated
- [ ] Failure cases in Appendix D
- [ ] Cumsum memory limitation stated

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
cp -r papers/_TEMPLATE papers/halo_regla
nano papers/halo_regla/variables.json
cd papers/halo_regla
python apply_vars.py --module halo_regla
grep -rn "{{" .  # verify empty
python build.py --module halo_regla --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE halo_regla
```

### مرحله ۲: اجرای HALO-ReGLA

```powershell
cd D:\SVD_Project
python run_all_4_halo_regla.py --models distilbert,roberta,gpt2 --gate_init_bias 1.8 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\halo_regla
python apply_vars.py --module halo_regla
python build.py --module halo_regla
