# Paper Specific Sections — zerokv_neo

## Methodology Description

ZeroKV-Neo: Outlier-Preserving KV Cache Pruning with Attention Sinks
Source implementation: `Codes/framework/compression_engine/zerokv_neo_compressor.py`.

Mathematical formulation:
$$
S = \mathrm{mean}_h(A_{h, :, :}), \quad \mathcal{I}_{sink} = \{0, \dots, s-1\}, \quad \mathcal{I}_{top} = \operatorname{topk}(S, k), \quad \mathcal{K} = \mathcal{I}_{sink} \cup \mathcal{I}_{top}
$$

## Results Narrative

- DistilBERT: 75.92% accuracy (PASS), 0.0% parameter change.
- RoBERTa: 50.92% accuracy (PASS_LOW), 0.0% parameter change.
- GPT-2: 51.15% accuracy (PASS_LOW), 0.0% parameter change.
