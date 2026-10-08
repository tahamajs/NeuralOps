# Pre-Submission Checklist — Performer

## Setup
- [ ] Copied `_TEMPLATE/` to `performer/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module performer`
- [ ] Verified no `[PLACEHOLDER]` remains

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Performer is a PUBLISHED method (mention Choromanski et al., 2021)
- [ ] Our contribution is the PORT, not the algorithm
- [ ] Two discrepancies disclosed (inject raise, output_scale)

### Introduction
- [ ] Motivates kernel attention
- [ ] Describes FAVOR+
- [ ] Discloses the two discrepancies

### Method
- [ ] Feature map detailed
- [ ] Numerical stabilization
- [ ] Both paths (bidi, causal)
- [ ] The two bugs analyzed

### Results
- [ ] All tables filled
- [ ] Feature ablation
- [ ] Kernel ablation
- [ ] LoRA rank ablation
- [ ] Recovery steps ablation

### Analysis
- [ ] Feature map analysis
- [ ] Kernel type comparison
- [ ] The two bugs
- [ ] Comparison with Cosformer

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Seeds documented
- [ ] Diagnostics included

## Appendices
- [ ] A: Math derivations
- [ ] B: Implementation details
- [ ] C: Full data
- [ ] D: Negative results
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render

## Final Pass
- [ ] Read aloud
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue
- [ ] Upload
```

---

## 📄 `build.py` and `apply_vars.py`

Same as previous templates.

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE performer
```

### مرحله ۲: پر کردن `variables.json`

### مرحله ۳: Apply و build

```powershell
cd papers\performer
python apply_vars.py --module performer
python build.py --module performer
```

---

## ⚠️ نکات مخصوص Performer — خلاصه ۹ کشف

### ۱. این **روش منتشر شده** است
Choromanski et al., ICLR 2021. در abstract صریح بگو که port هستی نه innovator.

### ۲. باگ کشنده — `inject_performer` raise می‌کند
```python
except Exception as e:
    logger.warning("...")
    raise RuntimeError(f"Injection failed")  # ← BUG
```
Docstring می‌گوید `returning None on failure` ولی code raise می‌کند.

### ۳. `output_scale` پارامتر جدید
```python
if not hasattr(self, "output_scale"):
    self.output_scale = nn.Parameter(torch.ones(1))
```
در checkpoint baseline وجود ندارد. `strict=True` می‌شکند.

### ۴. `omega` **persistent** است (خوب!)
```python
self.register_buffer("omega", omega, persistent=True)
```
برخلاف ORION که `persistent=False` داشت. RFF projection reload می‌شود.

### ۵. Numerical stabilization **خوب**
```python
max_proj = proj.amax(dim=-1, keepdim=True).detach()
diff = torch.clamp(proj - max_proj - diag_sq, min=-30.0, max=10.0)
```
Max-subtraction + clamp + eps. استاندارد.

### ۶. Chunk size = 64 (سریع)
```python
chunk_size = min(64, N)
```
برای N=128: فقط ۲ iteration. مثل ORION و بهتر از GatedDeltaNet (که ۸ داشت).

### ۷. Q_feat mask در encoder path
```python
Q_feat = Q_feat * mask_expanded
```
برای جلوگیری از تأثیر padding tokens. **خوب.**

### ۸. Cross-attention bypass
```python
if encoder_hidden_states is not None:
    return hidden_states, None
```
Cross-attention **پشتیبانی نمی‌شود** — ورودی بدون تغییر برمی‌گردد. این silent failure است.

### ۹. Feature map **فقط روی Q و K** اعمال می‌شود
V خام می‌ماند. این درست است.

---

## 📊 مقایسه با سایر ماژول‌های linear attention

| ماژول | Feature | Theory | Causal | Bug |
|---|---|---|---|---|
| Cosformer | cos/sin pos | Yes | Yes | No |
| **Performer** | **FAVOR+** | **Yes** | **Yes** | **raise** |
| ORION | RFF | Yes | Yes | raise + rff_proj |
| NASC | RFF (spectral) | No | Yes | raise in inject |

**Performer بهترین تئوری رو داره** ولی باگ raise داره.

---

**قدم بعدی: اگر نتایج Performer رو از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم.**