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
```

---

## 💡 نکات حیاتی

### ۱. صادقانه بگو FA3 روی RTX 4070 کار نمی‌کند

روی Ada Lovelace، FA3 kernel وجود ندارد. فقط SDPA fallback. این را 
در abstract و limitations بگو.

### ۲. Exactness — مزیت اصلی

تفاوت با Performer/Longformer/ClusterFormer: FA3 **exact** است. تمام 
outputها ریاضی معادل هستند.

### ۳. Not Compression

LoRA اضافه می‌کند. صادق باش.

### ۴. H100 Claims

اگر H100 نداری، claimهایت را extrapolated بنویس.

### ۵. FP8 Simulation

روی Ada، FP8 tensor cores نیست. فقط simulate می‌کنی.

---

**قدم بعدی: `run_all_4_flash3.py` را اجرا کن و نتایج را بفرست تا `variables.json` را پر کنم.**