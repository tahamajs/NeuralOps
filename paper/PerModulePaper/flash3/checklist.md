# Pre-Submission Checklist — FlashAttention-3 Wrapper

## Critical
- [ ] **Never call wrapper "compression"**
- [ ] Fallback to SDPA documented
- [ ] Exactness test provided
- [ ] Hardware requirements stated
- [ ] RTX 4070 uses SDPA (not FA3)
- [ ] Parameter increase disclosed

## Content
- [ ] Abstract under 250 words
- [ ] Dispatch logic documented
- [ ] Multi-architecture weight extraction
- [ ] FP8 support described
- [ ] Causal masking explained

## Honesty
- [ ] H100 benchmarks not claimed
- [ ] FP8 simulation disclosed
- [ ] Fallback path results reported
- [ ] Failure cases in Appendix D

## Quick Start
```bash
cp -r papers/_TEMPLATE papers/flash3
nano papers/flash3/variables.json
cd papers/flash3
python apply_vars.py --module flash3
python build.py --module flash3 --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱
```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE flash3
```

### مرحله ۲
```powershell
cd D:\SVD_Project
python run_all_4_flash3.py --models distilbert,roberta,gpt2 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴
```powershell
cd D:\SVD_Project\papers\flash3
python apply_vars.py --module flash3
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module flash3 --pdf
