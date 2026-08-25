# NeuralOps Attention Modules Library

This directory contains specialized attention mechanisms, linear recurrent layers, state-space models (SSMs), and factorized low-rank modules registered in `ATTENTION_REGISTRY`.

## 1. Top 10 Novel Attention Modules

| Module | File | Novel Idea | Complexity | T5-Compatible |
| :--- | :--- | :--- | :--- | :--- |
| `aegis_v6` | `aegis_v6.py` | SVD-factorized projections + energy thresholding + RFF global attention | $\mathcal{O}(L \cdot W + L \cdot F)$ | Yes |
| `mamba` | `mamba.py` | Bidirectional SSM state-space scan for encoder classification | $\mathcal{O}(L \cdot D)$ | Yes |
| `nasc_attention` | `nasc_attention.py` | SVD-LoRA factorized projections with rotary embeddings | $\mathcal{O}(L)$ | Yes |
| `spectron` | `spectron.py` | Low-rank sparse attention with Lagrangian spectral initialization | $\mathcal{O}(L \cdot r + L^2)$ | Yes |
| `phantom` | `phantom.py` | CUR matrix decomposition with leverage score pivot selection & LSH | $\mathcal{O}(L \cdot r + L \cdot B_s)$ | Yes |
| `reap_moe` | `reap_moe.py` | SVD + Mixture-of-Experts gated routing with load-balancing loss | $\mathcal{O}(L^2 + L \cdot k \cdot D)$ | Yes |
| `gated_delta_net` | `gated_delta_net.py` | Gated delta network with bidirectional associative recurrence | $\mathcal{O}(L \cdot d_k \cdot d_v)$ | Yes |
| `hgrn2_layer` | `hgrn2_layer.py` | Hierarchical gated recurrent network v2 with log-space decay | $\mathcal{O}(L \cdot D)$ | Yes |
| `salt_attention` | `salt_attention.py` | Singular value rescaling ($\gamma$) with additive low-rank adapter | $\mathcal{O}(L \cdot r + L^2)$ | Yes |
| `orion` | `orion.py` | Factorised linear with orthogonal RFF kernel & Softpick bifurcation | $\mathcal{O}(L \cdot (H_k F + H_f L))$ | Yes |

---

## 2. Registering a New Attention Module

To register a new custom attention module into the NeuralOps framework:

1. **Subclass `BaseAttention`**:
   Import `from framework.core.base import BaseAttention` and subclass it.
   ```python
   from framework.core.base import BaseAttention
   from framework.core.registry import ATTENTION_REGISTRY

   @ATTENTION_REGISTRY.register("my_novel_attention")
   class MyNovelAttention(BaseAttention):
       def __init__(self, config=None, rank=None, **kwargs):
           super().__init__(config, rank)
           self.d_model = getattr(config, "hidden_size", 768) if config else kwargs.get("hidden_size", 768)
   ```

2. **Implement Forward Pass Contract**:
   The `forward` method must accept standard kwargs and return a tuple `(output, attention_weights_or_None)`:
   ```python
       def forward(
           self,
           hidden_states: torch.Tensor,
           attention_mask: Optional[torch.Tensor] = None,
           head_mask: Optional[torch.Tensor] = None,
           output_attentions: bool = False,
           **kwargs: Any,
       ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
           # Process hidden_states [B, L, D] or [B, H, L, Dh]
           return output, None
   ```

3. **Provide Weight Adaptation (`copy_weights`)**:
   Implement a defensive `copy_weights(self, old_attn: Any)` method or rely on `BaseAttention.copy_weights` to transfer weights from standard `q_proj`, `k_proj`, `v_proj`, `out_proj` projections.

---

## 3. T5 Compatibility Requirements

T5 architectures and PyTorch inspection utilities expect sub-layer weights to expose a `.weight` property. Any custom factorized linear layers (such as SVD, LoRA, or CUR projections) must implement:

```python
@property
def weight(self) -> torch.Tensor:
    """Reconstructed effective weight matrix [out_features, in_features]."""
    return torch.matmul(self.up.weight, self.down.weight)
```

---

## 4. Forward Pass Interface Contract

Every attention layer in this directory adheres to the following unified interface:
- **Input Activations**: `hidden_states` of shape `[B, L, D]` (or `[B, H, L, Dh]`).
- **Attention Mask**: `attention_mask` tensor (2D `[B, L]`, 3D, or 4D broadcastable).
- **Return Type**: `Tuple[torch.Tensor, Optional[torch.Tensor]]` where output tensor is `[B, L, D]` and attention weights matrix is `None` or `[B, H, L, L]`.

---

## 5. Verification & Testing

Verify forward and backward passes across all registered attention modules:

```bash
cd Codes/framework
PYTHONPATH=../.. python3 verify_forward_passes.py
pytest tests/ -v --timeout=120
```
