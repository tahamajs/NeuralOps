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
