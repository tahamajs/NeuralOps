# Pre-Submission Checklist — REAP-MoE

## Setup
- [ ] Copied `_TEMPLATE/` to `reap_moe/`
- [ ] Filled `variables.json`
- [ ] Ran `python apply_vars.py --module reap_moe`
- [ ] No `[PLACEHOLDER]` remains

## Honesty
- [ ] Parameter **increase** is stated in abstract
- [ ] FFN duplication is acknowledged
- [ ] No "compression" claim for REAP-MoE
- [ ] Zero-init explained
- [ ] Aux loss weight reported
- [ ] Expert utilization shown

## Content
- [ ] MoE routing explained
- [ ] Load balancing derived
- [ ] SVD experts explained
- [ ] All ablation tables filled

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] Hyperparameters documented
- [ ] Seeds specified

## Final
- [ ] Read aloud
- [ ] PDF compiles
- [ ] External feedback

## Quick Start

```bash
cp -r papers/_TEMPLATE papers/reap_moe
nano papers/reap_moe/variables.json
cd papers/reap_moe
python apply_vars.py --module reap_moe
grep -rn "{{" .
python build.py --module reap_moe --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱: کپی

```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE reap_moe
```

### مرحله ۲: اجرا

```powershell
cd D:\SVD_Project
python run_all_4_reap_moe.py --models distilbert,roberta,gpt2 --n_experts 8 --top_k 2 --recovery_steps 50
```

### مرحله ۳-۵: پر کردن و ساخت

```powershell
cd D:\SVD_Project\papers\reap_moe
python apply_vars.py --module reap_moe
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module reap_moe
```

---

## 💡 نکات کلیدی REAP-MoE

### ⚠️ هشدار مهم — این یک compressor نیست

**REAP-MoE پارامترها را کم نمی‌کند. آن‌ها را زیاد می‌کند.**

در abstract صادق باش:
- ✅ «MoE increases capacity»
- ✅ «SVD reduces per-expert bloat»
- ❌ «REAP-MoE compresses model» (این غلط است)

### ۱. سه نوآوری REAP-MoE

1. **SVD experts** — هر expert با rank $d/4$ فشرده می‌شود
2. **Aux loss** — load balancing را تضمین می‌کند
3. **Zero-init** — training را پایدار می‌کند

### ۲. parameter growth

$E=8, r=d/4$ → params = $1.5 E d^2 = 12 d^2$ vs $8 d^2$ FFN اصلی = **۱.۵×**

پس REAP-MoE = **re-architecture** با capacity بیشتر، نه **compressor**.

### ۳. FFN Duplication

اگر framework فقط attention را جایگزین کند، FFN اصلی همچنان اجرا می‌شود. این باعث ۲× parameters و ۲× compute می‌شود. در `05_architecture.md` این را صادقانه بگو.

### ۴. expert utilization

اگر router collapse کند، MoE بی‌فایده است. **همیشه** $\rho_i$ هر expert را گزارش کن.

---

**قدم بعدی: بگو کدام ماژول را اول `variables.json` را با اعداد واقعی پر کنم. اگر `results/unified_reap_moe/*/summary.csv` را بفرستی، می‌توانم این کار را برایت انجام دهم.**