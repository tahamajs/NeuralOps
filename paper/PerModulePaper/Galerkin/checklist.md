# Pre-Submission Checklist — Galerkin Attention

## Critical
- [ ] **Never call Galerkin "compression"**
- [ ] Parameter increase disclosed
- [ ] **Encoder-only benefit documented**
- [ ] **Decoder fallback documented (GPT-2 uses standard softmax)**
- [ ] DCT basis explained
- [ ] Energy scaling documented

## Content
- [ ] Abstract under 250 words
- [ ] Continuous operator formulation
- [ ] Galerkin discretization explained
- [ ] DCT choice justified
- [ ] Encoder vs decoder dispatch

## Honesty
- [ ] GPT-2 no benefit documented
- [ ] Rank ablation present
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation

## Quick Start
```bash
cp -r papers/_TEMPLATE papers/galerkin
nano papers/galerkin/variables.json
cd papers/galerkin
python apply_vars.py --module galerkin
python build.py --module galerkin --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱
```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE galerkin
```

### مرحله ۲
```powershell
cd D:\SVD_Project
python run_all_4_galerkin.py --models distilbert,roberta,gpt2 --rank 64 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴
```powershell
cd D:\SVD_Project\papers\galerkin
python apply_vars.py --module galerkin
python build.py --module galerkin --pdf
