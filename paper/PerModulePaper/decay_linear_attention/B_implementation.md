# Appendix B. Implementation Details and Code Mapping

The `decay_linear_attention` module is implemented in `Codes/framework/attentions/decay_linear_attention.py` (782 lines).
- Registered Class: `LoRALinear`
- Base Architecture: `nn.Module`
- Constructor Signatures: `self, in\_features: int, out\_features: int, bias: bool = True, lora\_rank: int = 16, lora\_alpha: float = 32.0, freeze\_base: bool = False, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None, ) -> None: super().\_\_init\_\_() factory\_kwargs = {"device": device, "dtype": dtype} self.i`
