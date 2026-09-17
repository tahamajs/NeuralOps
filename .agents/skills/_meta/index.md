# Skill Index

## By Task

### I want to write a paper
1. `paper-writing/SKILL.md`
2. `tables/main-results.md`
3. `figures/rank-vs-accuracy.md`
4. `paper-writing/abstract-template.md`
5. `paper-writing/introduction-template.md`
6. `paper-writing/method-template.md`
7. `paper-writing/results-template.md`
8. `paper-writing/analysis-template.md`
9. `paper-writing/limitations-template.md`
10. `paper-writing/conclusion-template.md`

### I want to run ablations
1. `experiments/SKILL.md`
2. `experiments/rank-ablation.md`
3. `experiments/lora-ablation.md`
4. `experiments/recovery-ablation.md`
5. `experiments/calibration-ablation.md`

### I want statistical rigor
1. `metrics/statistical-significance.md`
2. `metrics/spectral-analysis.md`
3. `metrics/output-analysis.md`

### I want to publish checkpoints
1. `publishing/huggingface-upload.md`
2. `publishing/citation-bibtex.md`
3. `reproducibility/model-cards.md`

## By Dependency

| Skill | Depends on |
|---|---|
| `paper-writing/*` | `metrics/*`, `tables/*`, `figures/*` |
| `experiments/*` | trained models in `results/` |
| `metrics/spectral-analysis.md` | saved checkpoints |
| `metrics/attention-analysis.md` | forward hooks |
| `publishing/*` | completed, validated runs |
