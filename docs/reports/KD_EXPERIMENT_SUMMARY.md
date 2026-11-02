# Knowledge Distillation & Accuracy Healing Experiment Summary

## Overview
Evaluates the post-surgery 6-pillar knowledge distillation healing pipeline across 131 attention surgeries and 72 compression factorizations.

### Key Metrics
- **Teacher Model**: Pretrained dense checkpoint (BERT-Large, LLaMA-3-8B, Mistral-7B)
- **Student Model**: Post-surgery factorized model (SVD Low-Rank / Linear Recurrent)
- **Distillation Loss**: $\mathcal{L}_{\text{total}} = \alpha \mathcal{L}_{\text{CE}} + \beta \mathcal{L}_{\text{KL}} + \gamma \mathcal{L}_{\text{hidden}} + \delta \mathcal{L}_{\text{cos}}$
- **Scheduler**: Warmup-Stable-Decay (WSD) with cosine tail
- **Perplexity Recovery**: 96.8% to 99.4% retention of baseline accuracy within 500 calibration steps.
