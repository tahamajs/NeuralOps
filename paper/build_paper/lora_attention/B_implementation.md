# Appendix B. Implementation Details

## B.1 LoRALinear Module

```python
class LoRALinear(nn.Module):
    def __init__(
        self, in_features, out_features, bias=True, lora_rank=16, lora_alpha=32.0, freeze_base=False
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))

        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.empty(lora_rank, in_features))
            self.lora_B = nn.Parameter(torch.empty(out_features, lora_rank))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        self.reset_parameters()
        if freeze_base:
            self.freeze_base_weights()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.lora_rank > 0:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def freeze_base_weights(self):
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False
        if self.lora_rank > 0:
            self.lora_A.requires_grad = True
            self.lora_B.requires_grad = True

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            out = out + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.lora_scaling
        return out * self.output_scale
```

## B.2 Weight Extraction

```python
def copy_weights(self, src):
    extracted = {"q": (None, None), "k": (None, None), "v": (None, None), "o": (None, None)}
    
    # 1. UniversalAdapter
    for proj in ["q", "k", "v", "o"]:
        try:
            w, b = UniversalAdapter.get_weight_bias(src, proj)
            if w is not None:
                extracted[proj] = (w, b)
        except Exception as e:
            print(f'copy_weights failed: {e}')
            raise RuntimeError(f"copy_weights failed: {e}")
    
    # 2. GPT-2 Conv1D
    if hasattr(src, "c_attn") and (extracted["q"][0] is None or ...):
        weight = src.c_attn.weight.data
        if weight.shape[1] == 3 * self.hidden_size:
            weight = weight.t()
        q_w, k_w, v_w = weight.chunk(3, dim=0)
        extracted["q"] = (q_w, ...)
        ...
    
    # 3. RoBERTa nested
    if hasattr(src, "self"):
        s = src.self
        extracted["q"] = (s.query.weight, ...)
        ...
    
    # 4. Named projections
    named_map = {"q": ["q_proj", "q_lin", "query"], ...}
    
    # Nested detection
    if hasattr(src, "self") and hasattr(getattr(src, "self"), "query"):
        self.is_nested_encoder = True
    ...
```

## B.3 Common Pitfalls

### B.3.1 Silent Failure Corrupts Model

If copy_weights fails silently, the base weights stay at random init. 
LoRA Attention **raises exceptions** to prevent this.

### B.3.2 Wrong Nested Detection

Checking class name before `self.query` misclassifies DistilBERT.

### B.3.3 Forgetting to Freeze Base

If base weights train, you lose the "efficient" part of PEFT.

### B.3.4 Not Zero-Initializing B

If $B \neq 0$ at start, the model differs from baseline.

### B.3.5 Merge Precision Loss

FP16 merge introduces ~1e-3 error. For high-precision applications, merge 
in FP32.

---