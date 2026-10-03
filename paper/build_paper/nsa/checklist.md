# Pre-Submission Checklist — Native Sparse Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `nsa/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module nsa`
- [ ] No `[PLACEHOLDER]` remains

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
```

---

## 💡 نکات کلیدی NSA

### ۱. تفاوت اصلی با ASA-Sparse

| جنبه | ASA-Sparse | NSA |
|---|---|---|
| Local window | ندارد | **دارد** |
| Top-k | کل | **فقط خارج از window** |
| Density | کمتر | متوسط |
| پارامتر | sparsity_logits | **هیچ** |

### ۲. Union Mask

$$
\mathcal{S}_i = \mathcal{W}_i \cup \text{TopK}_{j \notin \mathcal{W}_i}(s_{ij})
$$

کلید **مهم:** top-k روی **خارج از window** محاسبه می‌شود، نه کل.

### ۳. ترتیب Mask مهم است

1. **اول** local window را mask کن
2. **بعد** causal mask (اگر decoder)
3. **بعد** top-k روی باقی‌مانده

اگر ترتیب اشتباه باشد، top-k ممکن است آینده‌ها را انتخاب کند.

### ۴. بدون پارامتر اضافه

NSA **صفر پارامتر** اضافه می‌کند. $w$ و $k$ **hyperparameter** هستند.

### ۵. Density مهم‌ترین معیار

$$
\rho = \frac{2w + 1 + k}{L}
$$

برای $L=128, w=32, k=16$: $\rho = 63\%$.
برای $L=2048$: $\rho = 4\%$.

**صادقانه:** NSA فقط **density** را کم می‌کند، نه زمان. برای speedup واقعی نیاز به **sparse kernels** داری.

### ۶. Short Sequence Handling

اگر $L \leq 2w+1$ باشد، همه چیز در window است و top-k **skip** می‌شود. این را ذکر کن.

### ۷. Results از سایر ماژول‌ها

از لاگ‌های قبلی:
- **DistilBERT:** ~89.5%
- **RoBERTa:** ~76.5%
- **GPT-2:** ~77.5%

این‌ها تخمینی هستند. اگر اجرا کرده‌ای، اعداد واقعی را وارد کن.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_nsa/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**