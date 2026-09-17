# Adversarial Robustness

## Purpose

Test whether compression hurts adversarial robustness.

## Attacks

- FGSM at ε = 0.001, 0.01, 0.05
- PGD at ε = 0.01, 10 steps
- Character-level random substitution

## Metrics

- Accuracy under attack
- Attack success rate

## Output

- `papers/modules/<module>/tables/adversarial.csv`
