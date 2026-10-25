---
name: neuralops-ablation-scientist
description: "Design, run, and log multi-pillar knowledge distillation ablation studies across regimes."
version: "1.0.0"
author: "NeuralOps Team"
tags: ["neuralops", "ablation", "distillation", "experimentation"]
---

# Skill: Ablation Scientist

## When to use
Trigger when executing systematic ablation experiments across knowledge distillation regimes and architectural variants.

## Prerequisites
- SSH access to `taha@192.168.1.100`
- `/mnt/d/venv/bin/python3`

## Steps
1. **Design 6-Pillar KD Ablation Matrix**:
   Configure the 6 distillation loss components:
   - Output logit KL divergence
   - Intermediate feature map MSE
   - Attention map Cosine similarity
   - SVD spectral singular value preservation loss
   - Hidden state covariance alignment
   - Contrastive representation loss
2. **Evaluate Regimes 0 through 5**:
   Execute training sweeps across regimes 0 (unweighted baseline) through 5 (full 6-pillar objective) for each candidate module.
3. **Log Run to Persistent Ledger**:
   Record telemetry to master ledger with `family="kd-6pillar-ablation"`.
4. **Generate Visualization Plots**:
   Plot Pareto frontier curves comparing downstream retention vs FLOPs reduction.

## Verification
- Confirm all 6 regimes produce valid evaluation checkpoints and loss convergence logs in the master ledger.

## Common pitfalls
- Tuning hyperparameters for only one regime, skewing comparative findings.
- Failing to set fixed random seeds across ablation regimes.
