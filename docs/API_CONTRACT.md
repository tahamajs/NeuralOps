# neuralops API Contract & User Interface Specification

## A. USER PERSONAS
- **Persona 1 (ML Researcher)**: Needs drop-in attention replacement with mathematical parity to reference architectures, transparent attention weight inspection, and minimal boilerplate.
- **Persona 2 (Knowledge Distillation Engineer)**: Needs layer-level KD hooks, memory-efficient fused losses, and flexible scheduling without writing custom trainers.
- **Persona 3 (Production ML Engineer)**: Needs high-throughput inference, TorchScript/ONNX export compatibility, low peak VRAM, and deterministic execution.
- **Persona 4 (Framework Developer)**: Needs standardized base classes, registry mechanisms, consistent constructor/forward signatures, and rigorous testing utilities.

---

## B. PUBLIC API SURFACE — TOP-LEVEL IMPORTS

### B.1 Top-Level Imports (`from neuralops import ...`)
```python
from neuralops import (
    GLAttention,
    CosformerAttention,
    SVDLinearAttention,
    DecayLinearAttention,
    create_attention,
    ATTENTION_REGISTRY,
    register_attention,
    count_parameters,
    get_attention_module,
    replace_attention,
)
```

### B.2 Submodule Import Equivalences
All import paths resolve to the identical class object:
```python
from neuralops.attentions import GLAttention as GLA1
from neuralops.attentions.linear import GLAttention as GLA2
from neuralops.attentions.linear.gla_attention import GLAttention as GLA3

assert GLA1 is GLA2 is GLA3
```

### B.3 No Wild Exports
All public modules define `__all__` explicitly.

### B.4 Versioning
```python
import neuralops
print(neuralops.__version__)      # e.g., '1.3.0'
print(neuralops.__api_version__)  # e.g., 'v1'
```

---

## C. ATTENTION LAYER CONTRACT

### C.1 Constructor Signature
```python
class GLAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        bias: bool = True,
        dropout: float = 0.0,
        is_causal: bool = True,
        max_seq_len: int = 2048,
        chunk_size: int = 64,
        tau: float = 1.0,
        use_chunked: bool = True,
        lora_rank: int = 0,
        lora_alpha: float = 32.0,
        freeze_base: bool = False,
        layer_idx: int = 0,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None: ...
```

### C.2 Forward Signature
```python
def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_ids: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, ...]] = None,
    **kwargs: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]: ...
```

### C.3 Return Value Contract
Exact return type hint: `Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]`.

### C.4 Weight Naming Contract
```
q_proj.weight, q_proj.bias
k_proj.weight, k_proj.bias
v_proj.weight, v_proj.bias
out_proj.weight, out_proj.bias
gate_proj.weight, gate_proj.bias (or gate_W1, gate_W2)
```

### C.5 Property Contract
Properties exposed: `query`, `key`, `value`, `dense`, `num_heads`, `head_dim`, `hidden_size`, `is_causal`, `supports_gqa`, `supports_bidirectional`.

### C.6 Weight Copy Contract
`copy_weights(src: nn.Module)` copies weights from GPT-2 (`c_attn`, `c_proj` Conv1D), LLaMA, BERT, RoBERTa.

### C.7 Injection Helper
`inject_gla(model: nn.Module, target_layers: Optional[List[int]] = None, copy_weights: bool = True, **attn_kwargs: Any) -> nn.Module`

---

## D. HF TRANSFORMERS INTEGRATION CONTRACT

### D.1 Config Class
```python
from transformers import PretrainedConfig

class GLAConfig(PretrainedConfig):
    model_type = "gla"

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        max_position_embeddings: int = 2048,
        chunk_size: int = 64,
        tau: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.chunk_size = chunk_size
        self.tau = tau
```

### D.2 Model Class
```python
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class GLAModel(PreTrainedModel):
    config_class = GLAConfig
    base_model_prefix = "gla"

AutoConfig.register("gla", GLAConfig)
AutoModel.register(GLAConfig, GLAModel)
AutoModelForCausalLM.register(GLAConfig, GLAModel)
```

### D.3 Save/Load Contract
```python
model.save_pretrained("/tmp/my_gla")
loaded = AutoModelForCausalLM.from_pretrained("/tmp/my_gla")
```

### D.4 trust_remote_code Example
```python
model = AutoModelForCausalLM.from_pretrained("username/gla-model", trust_remote_code=True)
```

### D.5 Generation Contract
```python
output = model.generate(input_ids, max_new_tokens=50, do_sample=False)
```

---

## E. KNOWLEDGE DISTILLATION CONTRACT

### E.1 Hook Registration Pattern
```python
s_outs, t_outs = [], []
for l in student.h:
    l.attn.register_forward_hook(lambda m, i, o: s_outs.append(o[0] if isinstance(o, tuple) else o))
for l in teacher.transformer.h[:len(student.h)]:
    l.attn.register_forward_hook(lambda m, i, o: t_outs.append(o[0] if isinstance(o, tuple) else o))
```

### E.2 Loss Function Signature + Example
```python
loss_fn = LayerAttentionDistillLoss(weight=0.5, loss_type="mse")
loss = loss_fn(s_outs, t_outs)
```

### E.3 Training Loop Integration (5-line example)
```python
# 1. Forward passes & record hooks
s_logits = student(x).logits; t_logits = teacher(x).logits
# 2. Compute losses
kd_loss = F.kl_div(F.log_softmax(s_logits / 2.0, dim=-1), F.softmax(t_logits / 2.0, dim=-1), reduction='batchmean') * 4.0
aux_loss = loss_fn(s_outs, t_outs) if sched.get_weight(step) > 0 else 0.0
# 3. Backward
total_loss = kd_loss + sched.get_weight(step) * aux_loss
total_loss.backward()
```

### E.4 Scheduler Signature + Example
```python
sched = LayerDistillScheduler(warmup_steps=2000, total_steps=10000, max_weight=0.5, final_weight=0.0)
w = sched.get_weight(step)
```

---

## F. FUSED KERNELS CONTRACT

### F.1 Function Signatures
```python
def fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor: ...

def fused_linear_kl_divergence(
    hidden: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_weight: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor: ...
```

### F.2 Numerical Parity Targets
Must match `F.cross_entropy` and `F.kl_div` to within `< 1e-4` relative error.

### F.3 Memory Savings Targets
$\ge 60\%$ reduction in peak allocated VRAM for $V=50000, M=4096$.

### F.4 Gradient Flow Requirements
Both `hidden` and `weight` are differentiable with exact backpropagation.

### F.5 Dtype Support
Full support for `torch.float32`, `torch.bfloat16`, `torch.float16`.

---

## G. BACKEND DISPATCH CONTRACT

### G.1 Usage
```python
from neuralops.core.backend import dispatch
out = dispatch("chunk_gla", q, k, v, g)
```

### G.2 Fallback Contract
Never raises `ImportError`. Falls back seamlessly: Triton $\to$ CUDA PyTorch $\to$ CPU PyTorch.

### G.3 Determinism Contract
Bit-identical across repeated runs with matching backend, $< 1e-4$ relative difference across distinct backends.

---

## H. ERROR HANDLING CONTRACT
- Base exception `NeuralOpsError` with specialized subclasses: `ConfigError`, `WeightCopyError`, `BackendError`, `NumericalError`.
- Clear diagnostic messages with contextual remedy suggestions.

---

## I. TYPE HINTS CONTRACT
Full strict type annotations conforming to `mypy --check-untyped-defs`.

---

## J. DOCUMENTATION CONTRACT
NumPy-style docstrings with parameter descriptions, mathematical formulations, and runnable examples.

---

## K. BACKWARD COMPATIBILITY CONTRACT
Legacy imports and aliases maintained via backward-compatibility modules.

---

## L. TESTING CONTRACT
Unit tests, integration tests, numerical stability tests, and doctests targeting $> 85\%$ coverage on core modules.

---

## M. INTERACTION FLOWS

### M.1 Flow 1: Replace GPT-2 Attention
```python
import torch
from transformers import GPT2LMHeadModel
from neuralops.attentions.linear import GLAttention
from neuralops.utils import replace_attention

model = GPT2LMHeadModel.from_pretrained("gpt2")
replace_attention(model, new_attention_cls=GLAttention, hidden_size=768, num_heads=12, head_dim=64, copy_weights=True)
input_ids = torch.tensor([[15496, 995]])
out = model(input_ids)
assert out.logits.shape == (1, 2, 50257)
```

### M.2 Flow 2: Distillation Loop
Full runnable 5-line integration with `LayerAttentionDistillLoss` and `LayerDistillScheduler`.

### M.3 Flow 3: Production ONNX Export
Validates standard model export without graph partition breaks.

### M.4 Flow 4: Custom Registry Extension
Registering custom attention variant into `ATTENTION_REGISTRY` and retrieving with `create_attention`.

---

## N. VALIDATION CHECKLIST
- [x] B.1 — Top-level imports work
- [x] B.2 — Submodule imports work
- [x] B.3 — `__all__` complete
- [x] C.1 — Constructor signature matches
- [x] C.2 — Forward signature matches
- [x] C.3 — Return tuple correct
- [x] C.4 — Weight naming consistent
- [x] C.5 — Properties exist
- [x] C.6 — copy_weights works
- [x] C.7 — inject_gla works
- [x] D.1 — Config registered
- [x] D.2 — Model registered
- [x] D.3 — Save/load works
- [x] D.4 — trust_remote_code works
- [x] D.5 — generate works
- [x] E.1 — Hooks fire
- [x] E.2 — Loss works
- [x] E.3 — Loop integration works
- [x] E.4 — Scheduler works
- [x] F.1 — Fused signatures match
- [x] F.2 — Numerical parity
- [x] F.3 — Memory savings
- [x] F.4 — Gradients flow
- [x] F.5 — All dtypes
- [x] G.1 — Dispatch works
- [x] G.2 — Fallback works
- [x] G.3 — Determinism
- [x] H.1 — Exceptions defined
- [x] H.2 — Error messages complete
- [x] H.3 — Warnings vs errors
- [x] I.1 — Type hints
- [x] I.2 — mypy passes
- [x] J.1-J.5 — Docstrings
- [x] K.1 — Old imports work
- [x] K.2 — Old config names work
- [x] K.3 — Deprecation protocol
- [x] L.1 — Test coverage
- [x] M.1 — Flow 1 runs
- [x] M.2 — Flow 2 runs
- [x] M.3 — Flow 3 runs
- [x] M.4 — Flow 4 runs

---

## O. DESIGN DECISIONS
All architectural decisions referenced in `docs/API_DECISIONS.md`.
