# Pre-Submission Checklist — FUSIONATTN

## Critical
- [ ] **Never call FUSIONATTN "compression"**
- [ ] Parameter increase disclosed
- [ ] **Short-path silencing documented** (L=128 uses short path)
- [ ] Long-path untested — disclosed
- [ ] `raise RuntimeError` in inject documented
- [ ] Block gating explained

## Content
- [ ] Abstract under 250 words
- [ ] Dispatch logic documented
- [ ] Gate MLP architecture
- [ ] Top-K selection
- [ ] Block 0 preservation
- [ ] All ablations present

## Honesty
- [ ] Short path silent at L=128 — acknowledged
- [ ] Long path untested — acknowledged
- [ ] Failure cases in Appendix D
- [ ] Single-seed limitation

## Quick Start
```bash
cp -r papers/_TEMPLATE papers/fusion_attn
nano papers/fusion_attn/variables.json
cd papers/fusion_attn
python apply_vars.py --module fusion_attn
python build.py --module fusion_attn --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱
```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE fusion_attn
```

### مرحله ۲
```powershell
cd D:\SVD_Project
python run_all_4_fusion_attn.py --models distilbert,roberta,gpt2 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴
```powershell
cd D:\SVD_Project\papers\fusion_attn
python apply_vars.py --module fusion_attn
python build.py --module fusion_attn --pdf
```

---

## 💡 نکات حیاتی FUSIONATTN

### ۱. Short-Path Problem — مهم‌ترین نکته

روی `L=128`، شرط `if N <= 128` **همیشه True** است. پس:
- Long path هرگز اجرا نمی‌شود
- Gate MLP هرگز استفاده نمی‌شود
- Sparsity هرگز فعال نمی‌شود
- نتیجه = LoRA-only

این را **صادقانه** در abstract و limitations بگو.

### ۲. Threshold = 128

اگر threshold را به 64 کاهش بدهی، long path روی L=128 فعال می‌شود. توصیه می‌کنم یک ablation اضافه کنی.

### ۳. باگ `raise RuntimeError`

در `inject_fusion_attn`:
```python
except Exception:
    raise RuntimeError("copy_weights failed")
```

این باید `return None` باشد. وگرنه با هر خطا، injection می‌ترکد.

### ۴. `output_scale` اضافه

در `LoRALinear`، `output_scale` ساخته می‌شود ولی ممکن است در checkpoint قدیمی نباشد.

### ۵. Long-context فقط

FUSIONATTN برای `L ≥ 256` طراحی شده. برای SST-2 روی L=128، **benefit صفر است**.

### ۶. مقایسه صادقانه

| روش | در L=128 | در L=512 |
|---|---|---|
| Adasplash | فعال (α-entmax) | فعال |
| ClusterFormer | فعال (cluster) | فعال |
| **FUSIONATTN** | **غیرفعال (short)** | فعال |

---

**قدم بعدی: `run_all_4_fusion_attn.py` را اجرا کن و نتایج را بفرست تا `variables.json` را پر کنم. اگر خواستی threshold را به 64 تغییر بدهیم تا long path فعال شود.**