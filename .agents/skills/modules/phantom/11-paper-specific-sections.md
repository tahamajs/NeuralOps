# Paper Specific Sections — phantom

## Methodology Description

Phantom: Parameter-Hashed Attention with Token-wise Orthogonal Mapping
Source implementation: `Codes/framework/attentions/phantom.py`.

Mathematical formulation:
$$
W \approx C U^+ R, \quad A_{i,j} = \operatorname{Softpick}\left(\frac{Q K^\top}{\sqrt{d}} \odot M_{LSH}\right)
$$

## Results Narrative

- DistilBERT: 72.59% accuracy (PASS), -18.50% parameter change.
- RoBERTa: 50.92% accuracy (PASS_LOW), -14.19% parameter change.
- GPT-2: 50.92% accuracy (PASS_LOW), -19.91% parameter change.
