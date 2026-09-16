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
```

---

## 💡 نکات کلیدی Primal Attention

### ۱. تفاوت اصلی با Nyströmformer

**Nyströmformer:** landmarks مشترک بین $Q$ و $K$.
**Primal:** landmarks مجزا → **asymmetric**.

این را در abstract و method **برجسته** کن.

### ۲. سه فاکتور به جای یک

Primal attention matrix را به سه فاکتور تجزیه می‌کند:

$$
A \approx F_1 F_2^{-1} F_3
$$

که $F_1 \in \mathbb{R}^{L \times m}$، $F_2 \in \mathbb{R}^{m \times m}$، $F_3 \in \mathbb{R}^{m \times L}$.

### ۳. Schulz به جای SVD

Pseudoinverse با SVD محاسبه می‌شود و بعد با Schulz refinement بهبود می‌یابد:

$$
X_{t+1} = X_t (2I - F_2 X_t)
$$

**مهم:** `X_0 = pinv(F_2)` (نه صفر).

### ۴. بدون کاهش پارامتر

Primal **پارامتر کم نمی‌کند** — فقط attention memory را.

| جنبه | Primal |
|---|---|
| Parameters | تغییر نمی‌کند |
| Attention memory | **{{MEM_REDUCTION}}× کاهش** |
| Inference time (L=128) | کمی کندتر |
| Inference time (L>512) | **سریع‌تر** |

### ۵. Associativity برای خطی بودن

نکته کلیدی برای خطی بودن:

$$
\tilde{A} V = F_1 (F_2^{-1} (F_3 V))
$$

اگر اشتباهاً $\tilde{A}$ را مستقیماً محاسبه کنی، **هنوز $\mathcal{O}(L^2)$ است**.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_primal/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**