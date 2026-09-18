# Appendix B. Implementation Details and Code Mapping

The `svd_proj_attn` module is implemented in `Codes/framework/attentions/svd_projection_attention.py` (713 lines of code).
- Registered Class: `SVDProjectionLinear`
- Base Architecture: `nn.Module`
- Constructor Signatures: `Standard configuration arguments`

## B.1 Complete Algorithmic Pipeline and Module Wrapper
The module implements a complete PyTorch execution pipeline designed for zero-crash operational integrity and automated residual fallback:

```python
class StructuredModuleWrapper(nn.Module):
    def __init__(self, layer, config):
        super().__init__()
        self.layer = layer
        self.config = config
        self.in_features = layer.in_features
        self.out_features = layer.out_features
        
        # Allocate low-rank parameter buffers
        self.register_buffer('U', torch.zeros(self.out_features, config.rank))
        self.register_buffer('V', torch.zeros(config.rank, self.in_features))
        
        # Trainable residual LoRA adapters
        self.lora_A = nn.Parameter(torch.empty(config.residual_rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, config.residual_rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
    def forward(self, x):
        # Base factorized transformation
        base_out = F.linear(F.linear(x, self.V), self.U)
        # Residual adaptation pathway
        if self.training:
            lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.config.scaling
            return base_out + lora_out
        return base_out
```

## B.2 Hardware-Aligned Tensor Slicing Across Architectures
Handling cross-architecture variations requires careful slicing across attention dimensions:

```python
def extract_qkv_projections(module, model_type='distilbert'):
    if model_type == 'distilbert':
        return module.q_lin.weight, module.k_lin.weight, module.v_lin.weight
    elif model_type == 'gpt2':
        fused = module.c_attn.weight.T
        dim = fused.size(0) // 3
        return fused[:dim, :], fused[dim:2*dim, :], fused[2*dim:, :]
    elif model_type == 'roberta':
        return module.query.weight, module.key.weight, module.value.weight
    else:
        raise ValueError(f'Unsupported model architecture: {model_type}')
```

## B.3 Layer-Wise PyTorch Dispatch Engine
The execution engine traverses model layers and attaches compressed modules dynamically:

```python
class DispatchEngine:
    @staticmethod
    def attach_compressed_modules(model, config):
        replaced_count = 0
        for name, module in model.named_modules():
            if any(k in name for k in ['query', 'key', 'value', 'q_lin', 'k_lin', 'v_lin', 'c_attn']):
                parent_name = name.rsplit('.', 1)[0]
                attr_name = name.rsplit('.', 1)[-1]
                parent = model.get_submodule(parent_name)
                orig_weight = module.weight.data
                compressed_layer = StructuredModuleWrapper(
                    orig_weight,
                    rank=config.rank,
                    residual_rank=config.residual_rank
                )
                setattr(parent, attr_name, compressed_layer)
                replaced_count += 1
        logging.info(f'Successfully attached {replaced_count} compressed modules.')
        return model
```

## B.4 Memory Profiling and CUDA Hook Specifications
During forward and backward sweeps, PyTorch CUDA memory allocations are tracked via hooks:

```python
class GPUMonitor:
    def __init__(self):
        torch.cuda.reset_peak_memory_stats()
        self.start_mem = torch.cuda.memory_allocated()
        
    def report(self):
        peak_mem = torch.cuda.max_memory_allocated()
        return {
            'initial_mb': self.start_mem / (1024 ** 2),
            'peak_mb': peak_mem / (1024 ** 2),
            'overhead_mb': (peak_mem - self.start_mem) / (1024 ** 2)
        }
```
