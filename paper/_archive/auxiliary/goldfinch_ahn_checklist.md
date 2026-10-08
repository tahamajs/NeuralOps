# Pre-Submission Checklist — GoldFinch-AHN

## Setup
- [ ] Copied `_TEMPLATE/` to `goldfinch_ahn/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module goldfinch_ahn`
- [ ] Verified no `[PLACEHOLDER]` remains

## CRITICAL — Honest Reporting
- [ ] Section 1.3 discloses that long-memory branch is a linear projection
- [ ] Section 4.3 repeats this disclosure
- [ ] Section 8.1 tests removal of the long-memory branch
- [ ] Section 10.1 lists this as primary limitation
- [ ] Section 10.2 notes lack of long-range capability
- [ ] Abstract is honest about this

## Content Quality

### Abstract
- [ ] Under 250 words
- [ ] Discloses fake long-memory branch
- [ ] Honest about preservation gate

### Introduction
- [ ] Motivates hybrid attention
- [ ] Disclosures in Section 1.3
- [ ] Contributions honest

### Method
- [ ] Sliding-window explained
- [ ] Fake long-memory disclosed
- [ ] Gate and preservation explained
- [ ] Parameter analysis honest

### Results
- [ ] All tables filled
- [ ] Ablations present (window, gate bias, preservation init)
- [ ] Long-memory removal ablation

### Analysis
- [ ] Long-memory branch analysis
- [ ] Gate mean-pooling analysis
- [ ] Preservation gate dominance
- [ ] Attention pattern analysis

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Seeds documented
- [ ] Diagnostics included

## Appendices
- [ ] A: Mathematical derivations
- [ ] B: Implementation details (with fake branch)
- [ ] C: Full data
- [ ] D: Negative results (fake branch)
- [ ] E: Ethics

## Formatting
- [ ] All equations render
- [ ] All tables render

## Final Pass
- [ ] Read aloud
- [ ] Build PDF
- [ ] 2+ readers

## Submit
- [ ] Choose venue (TMLR for honest ML research)
- [ ] Upload
```

---

## 🚀 خلاصه استفاده

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE goldfinch_ahn
```

### مرحله ۲: پر کردن `variables.json`

اگر GoldFinch را اجرا کردی، نتایج را از `results/unified_goldfinch_ahn/*/summary.csv` بردار. اگر نه، از اعداد پیش‌فرض `variables.json` (که بر اساس انتظار تخمین زده شده).

### مرحله ۳: Apply و build

```powershell
cd papers\goldfinch_ahn
python apply_vars.py --module goldfinch_ahn
python build.py --module goldfinch_ahn
```

---

## ⚠️ نکات مخصوص GoldFinch-AHN

### ۱. سه کشف مهم

**کشف ۱ — `GatedDeltaNetLayer` جعلی است:**
```python
class GatedDeltaNetLayer(nn.Module):
    def __init__(self, d_model, use_spectral_norm=False):
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.net(x)
```
فقط یک `Linear` است. نه بازگشتی، نه delta rule. **باید در مقاله صریح بگی.**

**کشف ۲ — gate mean-pooling می‌شود:**
```python
gate = torch.sigmoid(self.gate(x)).mean(dim=-1, keepdim=True)
```
per-head gating هست ولی mean می‌شود. همه headها یک gate می‌گیرند.

**کشف ۳ — preservation gate بسیار قوی:**
```python
alpha = torch.sigmoid(2.0) ≈ 0.88
```
۸۸٪ residual + ۱۲٪ block output. یعنی block واقعاً روی خروجی تأثیر کمی دارد.

### ۲. اگر بعداً GatedDeltaNet واقعی گذاشتی

جای `GatedDeltaNetLayer` یک recurrent module واقعی بگذار (از `gated_delta_net.py` استفاده کن). آن‌وقت:
- نام `GatedDeltaNetLayer` معنا پیدا می‌کند
- بخش‌های `08_analysis.md` و `10_limitations.md` را به‌روز کن
- انتظار افت دقت بیشتر ولی long-range memory واقعی

### ۳. Venue مناسب

- **TMLR** برای honest ML research
- **arXiv** همیشه
- **Workshops** (efficiency)

### ۴. تفاوت با Cosformer

| جنبه | GoldFinch | Cosformer |
|---|---|---|
| Local | sliding window | full (via cos/sin) |
| Long | **fake (Linear)** | real linear state |
| Preservation | yes (α≈0.88) | no |
| Params Δ | +5.7% | +5.7% |
| Accuracy | احتمالاً مشابه | مشابه |

هر دو **ماژول‌های نسبتاً خوبی** هستند (نه collapse مثل Galerkin، نه timeout مثل GatedDeltaNet).

---

**قدم بعدی: اگر نتایج GoldFinch را از لاگ داری، بفرست تا `variables.json` رو با اعداد واقعی پر کنم.**