# Pre-Submission Checklist — Native Hybrid Attention

## Setup
- [ ] Copied `_TEMPLATE/` to `nha/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module nha`
- [ ] No `[PLACEHOLDER]` remains

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
```

---

## 💡 نکات کلیدی NHA

### ۱. تفاوت اصلی با HALO-ReGLA

| جنبه | HALO-ReGLA | NHA |
|---|---|---|
| Local branch | Global dense | **Local window** |
| Global branch | ELU+1 linear | **RFF orthogonal** |
| Gate type | Per-head | **Token-level** |
| Gate init | 1.8 | **3.0** |
| Cumsum | Full | **Chunked state** |

### ۲. دو استریم موازی

- **Local:** پنجره sliding برای pattern محلی
- **Global:** RFF برای context دور

ترکیب با گیت token-level:

$$
O_i = g_i \cdot O_i^\text{local} + (1 - g_i) \cdot O_i^\text{global}
$$

### ۳. RFF با ماتریس Orthogonal

بلوک‌های QR به جای Gaussian i.i.d.:

$$
\Omega = [Q_1, Q_2, \ldots] \cdot \sqrt{d_h}
$$

مزیت: variance کمتر، training پایدارتر.

### ۴. Causality با Chunked Prefix Scan

به جای cumsum کامل، state $S, Z$ را بین chunkها حمل می‌کنیم:

$$
S \leftarrow S + \phi(K_c)^\top V_c
$$

**مزیت:** memory $\mathcal{O}(m d_h)$ به جای $\mathcal{O}(L d_h^2)$.

### ۵. نتایج واقعی از لاگ

از لاگ‌های تو:

| Model | NHA | Baseline | Δ |
|---|---|---|---|
| DistilBERT | **90.25%** | 90.71% | -0.46% |
| RoBERTa | **FAIL** | 78.56% | (OOM در 12 لایه) |
| GPT-2 | **78.78%** | 79.36% | -0.58% |

**صادقانه:** RoBERTa OOM می‌دهد. این را در abstract و limitations ذکر کن.

### ۶. Gate Bias = 3.0

مقدار اولیه:

$$
\sigma(3) \approx 0.95
$$

یعنی ۹۵٪ local، ۵٪ global. مدل از behavior pretrained شروع می‌کند.

### ۷. L1 Regularization ندارد

برخلاف ASA-Span، NHA **L1 penalty** ندارد. گیت آزادانه یاد می‌گیرد.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_nha/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**