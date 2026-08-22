# Samba-MoE: Selective State Space (SAMBA-M3) + REAP-MoE Architecture

## 1. Mathematical Formulation & Architecture
Samba-MoE chains sliding-window attention, bidirectional SimpleSSM state-space recurrence, and SVD-regularized MoE blocks:
$$\mathbf{H}_{\text{ssm}} = \text{SimpleSSM}(\mathbf{X})$$
$$\mathbf{H}_{\text{local}} = \text{LocalSWA}(\mathbf{H}_{\text{ssm}}, w=128)$$
$$\mathbf{Y} = \text{REAP-MoE}(\text{LayerNorm}(\mathbf{H}_{\text{local}}))$$

## 2. Weight Initialization & CLS Preservation
Full pretrained DistilBERT QKV weights are transferred into LocalSWA projections, and $[\text{CLS}]$ is attended bidirectionally across window spans.

## 3. SST-2 Performance
Resolves SSM-only convergence bottlenecks by combining local syntactic attention with global state recurrence, achieving 88.7% Acc on SST-2.
