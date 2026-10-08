# Pre-Submission Checklist — NEXUS

## Critical Honesty Checks
- [ ] **Stripe inactive at N=128 documented**
- [ ] **RFF inactive by default documented**
- [ ] **Causal decoder uses dense fallback documented**
- [ ] **Only 4 of 6 components active documented**

## Setup
- [ ] Copied `_TEMPLATE/` to `nexus/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module nexus`
- [ ] No `[PLACEHOLDER]` remains

## Abstract
- [ ] Under 250 words
- [ ] Six components mentioned
- [ ] Caveats explicit
- [ ] Modular framework positioning

## Method
- [ ] All six components formulaically defined
- [ ] Activation conditions documented
- [ ] Complexity analysis

## Results
- [ ] Main results with active path
- [ ] Component ablation
- [ ] Block size ablation
- [ ] Top-K ablation
- [ ] Softpick temperature
- [ ] Component activation summary

## Limitations
- [ ] Stripe inactive
- [ ] RFF inactive
- [ ] Dense fallback
- [ ] Softpick risk
- [ ] copy_weights raises

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/nexus
nano papers/nexus/variables.json
cd papers/nexus
python apply_vars.py --module nexus
grep -rn "{{" .
python build.py --module nexus --pdf
```
```

---

## 🎯 خلاصه نهایی

### نکات حیاتی برای مقاله

| هشدار | چرا مهم است |
|---|---|
| **Stripe فقط $N > 256$** | در SST-2 (128) غیرفعال |
| **RFF فقط با `enable_spectraformer=True`** | پیش‌فرض False |
| **Causal → dense** | GPT-2 هیچ sparsity نمی‌گیرد |
| **Softpick در all-negative → uniform** | failure mode |
| **4 از 6 component فعال** | ادعا با واقعیت فرق دارد |
| **`raise` در copy_weights** | fix شود |

### نوآوری‌های قابل برجسته‌سازی

1. **Modular framework** — امکان enable/disable هر component
2. **Combination study** — اولین ترکیب شش تکنیک ۲۰۲۵
3. **Multi-architecture** — ۴ معماری
4. **Honest documentation** — صراحت دربارهٔ component فعال

### تفاوت با سایر مقالات

| مقاله | نوآوری اصلی |
|---|---|
| A3/AASVD | Weight compression |
| TCA | Token compression |
| Linformer | Sequence projection |
| LEKAv3 | Learned landmarks |
| Mamba SSM | SSM-inspired |
| MoE-DeltaKV | MoE + simulation |
| **NEXUS** | **6-component combination** |

---

**قدم بعدی: `run_all_4_nexus.py` را اجرا کن، اعداد را بردار، و در `variables.json` بگذار. اگر می‌خواهی fix `copy_weights` را انجام دهم، بگو.**