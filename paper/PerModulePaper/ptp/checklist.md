# Pre-Submission Checklist — Progressive Token Pruning

## Setup
- [ ] Copied `_TEMPLATE/` to `ptp/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module ptp`
- [ ] No `[PLACEHOLDER]` remains

## Content
- [ ] Abstract under 250 words
- [ ] Score network described
- [ ] [CLS] preservation explained
- [ ] Causal handling described

## Method
- [ ] Score formula numbered
- [ ] Top-k formula numbered
- [ ] [CLS] proof in appendix
- [ ] Causal proof in appendix

## Results
- [ ] Main table with kept ratio
- [ ] Keep ratio ablation
- [ ] [CLS] ablation
- [ ] Score network hidden ablation
- [ ] Token score analysis

## Honesty
- [ ] Score network sees future tokens (limitation stated)
- [ ] Shared selection across queries (limitation stated)
- [ ] Very short sequences limitation
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
cp -r papers/_TEMPLATE papers/ptp
nano papers/ptp/variables.json
cd papers/ptp
python apply_vars.py --module ptp
grep -rn "{{" .  # verify empty
python build.py --module ptp --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE ptp
```

### مرحله ۲: اجرای PTP

```powershell
cd D:\SVD_Project
python run_all_4_ptp.py --models distilbert,roberta,gpt2 --keep_ratio 0.8 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴: Apply و ساخت

```powershell
cd D:\SVD_Project\papers\ptp
python apply_vars.py --module ptp
python build.py --module ptp
```

---

## 💡 نکات کلیدی PTP

### ۱. تفاوت اصلی با سایر ماژول‌ها

| ماژول | چه چیزی prune می‌کند | per-query؟ |
|---|---|---|
| ASA-Sparse | top-k per query | ✅ |
| NSA | top-k per query (خارج از window) | ✅ |
| **PTP** | **top-k tokens (shared)** | ❌ |

**مهم:** PTP **توکن‌ها** را حذف می‌کند، نه اتصالات. یک مجموعه T برای همه queryها.

### ۲. Score Network

```python
nn.Sequential(
    nn.Linear(d, 64),
    nn.GELU(),
    nn.Linear(64, 1),
)
```

+ sigmoid → score در (0, 1).

### ۳. [CLS] Preservation — حیاتی

```python
scores[:, 0] = 1e4  # قبل از topk
```

**بدون این، accuracy سقوط می‌کند.**

### ۴. ترتیب Operations

1. Score
2. **CLS preservation** (score=1e4)
3. **Padding mask**
4. **CLS preservation دوباره** (padding mask ممکن است overwrite کند)
5. Top-k
6. Sort indices

### ۵. Causality

بعد از pruning، برای مقایسه causality از **original positions** استفاده کن:

```python
k_orig_pos = top_indices.unsqueeze(1).unsqueeze(2)
causal_mask = k_orig_pos > q_pos
```

### ۶. GQA Order

**قبل از** pruning، `repeat_kv` بزن:

```python
if num_heads != num_kv_heads:
    k = repeat_kv(k, num_heads // num_kv_heads)
    v = repeat_kv(v, num_heads // num_kv_heads)
```

### ۷. محدودیت‌های صادقانه

- **Score network آینده را می‌بیند** — برای generation مناسب نیست
- **Shared selection** — per-query نیست
- **Short sequences** ($L < 32$) — افت شدید
- **میانگین speedup** فقط ~1.25× در $\rho=0.8$

### ۸. نتایج تخمینی

| مدل | PTP | Baseline | Δ |
|---|---|---|---|
| DistilBERT | ~89.5% | 90.71% | -1.2% |
| RoBERTa | ~77.0% | 78.56% | -1.6% |
| GPT-2 | ~78.0% | 79.36% | -1.4% |

این‌ها تخمینی هستند — از `results/unified_ptp/*/summary.csv` اعداد واقعی را وارد کن.

---

**قدم بعدی: `variables.json` را با اعداد واقعی از `results/unified_ptp/*/summary.csv` پر کن. اگر خواستی من این کار را برایت انجام دهم — فقط اعداد CSV را بفرست.**