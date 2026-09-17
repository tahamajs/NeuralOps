# 5. Architecture and Implementation

## 5.1 SVDLinear Module
Dense layers selected for compression are substituted with `SVDLinear`:
$$y = (x V^\top) U^\top + \text{bias}$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py lines 74, 188-191 -->

When optional recovery adaptation is enabled, auxiliary LoRA parameters are evaluated in parallel:
$$y = (x V^\top) U^\top + \frac{\alpha}{r_{\text{lora}}} (x A_{\text{lora}}^\top) B_{\text{lora}}^\top + \text{bias}$$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 74 -->

## 5.2 Module Parameters
- `V`: `nn.Parameter` of shape $(r, d_{\text{in}})$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 105 -->
- `U`: `nn.Parameter` of shape $(d_{\text{out}}, r)$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 106 -->
- `bias`: `nn.Parameter` of shape $(d_{\text{out}})$ <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 108 -->
- `lora_A`: `nn.Parameter` of shape $(r_{\text{lora}}, d_{\text{in}})$ (optional) <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 111 -->
- `lora_B`: `nn.Parameter` of shape $(d_{\text{out}}, r_{\text{lora}})$ (optional) <!-- source: Codes/framework/compression_engine/aa_svd_compressor.py line 112 -->
