# Pre-Submission Checklist — ClusterFormer

## Critical
- [ ] **Never call ClusterFormer "compression"**
- [ ] Parameter increase clearly stated
- [ ] Complexity analysis correct ($O(KL)$)
- [ ] Causal decoder path uses exact attention (documented)
- [ ] Crossover point mentioned ($L \approx 100$)

## Content
- [ ] Abstract under 250 words
- [ ] K-means algorithm documented
- [ ] Centroid-to-key formula clear
- [ ] Blend mechanism explained
- [ ] All ablations (K, W, iters, τ, α) present

## Analysis
- [ ] Cluster quality metrics
- [ ] FLOPs comparison at multiple sequence lengths
- [ ] Comparison with Performer/Longformer/Reformer
- [ ] Failure cases in Appendix D

## Reproducibility
- [ ] Code on GitHub
- [ ] Checkpoints on HF
- [ ] K-means implementation documented
- [ ] Seeds documented

## Quick Start
```bash
cp -r papers/_TEMPLATE papers/clusterformer
nano papers/clusterformer/variables.json
cd papers/clusterformer
python apply_vars.py --module clusterformer
grep -rn "{{" .
python build.py --module clusterformer --pdf
```
```

---

## 🚀 خلاصه اجرا

### مرحله ۱
```powershell
cd D:\SVD_Project\papers
Copy-Item -Recurse _TEMPLATE clusterformer
```

### مرحله ۲
```powershell
cd D:\SVD_Project
python run_all_4_cluster.py --models distilbert,roberta,gpt2 --num_clusters 16 --window_size 32 --recovery_steps 50
```

### مرحله ۳: پر کردن `variables.json`

### مرحله ۴
```powershell
cd D:\SVD_Project\papers\clusterformer
python apply_vars.py --module clusterformer
Select-String -Path "*.md" -Pattern "\{\{"
python build.py --module clusterformer --pdf
```

---

## 💡 نکات مخصوص ClusterFormer

### ۱. برجسته‌سازی Linear بدون Random Features

تفاوت اصلی با Performer: ClusterFormer **exact** است روی سطح centroid. این را در abstract و method تأکید کن.

### ۲. Crossover Point

ClusterFormer برای $L < 100$ کندتر است. این را صادقانه در discussion بگو.

### ۳. Causal Decoder Fallback

ClusterFormer برای decoder از exact attention استفاده می‌کند. این محدودیت را باید توضیح بدهی.

### ۴. K-Means Quality

Cluster assignments باید visualize شوند. Cluster entropy و effective K را گزارش کن.

### ۵. مقایسه با `routing` Module

`routing` (که قبلاً دیدیم) از cluster-by-cluster loop استفاده می‌کند (کند). ClusterFormer از centroid-to-key استفاده می‌کند (سریع). این تفاوت را برجسته کن.

### ۶. Not Compression

مثل NHA و AdaSplash، ClusterFormer پارامتر اضافه می‌کند. صادق باش.

---

**قدم بعدی: `run_all_4_cluster.py` را اجرا کن و نتایج را بفرست تا `variables.json` را با اعداد واقعی پر کنم.**