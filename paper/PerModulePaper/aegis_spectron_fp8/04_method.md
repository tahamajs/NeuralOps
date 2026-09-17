# 4. Method Formulation
1. **FP8 Dynamic Simulation**:
   $$	ilde{x} = \operatorname{clamp}\left(\operatorname{round}\left(x \cdot rac{M_{	ext{fp8}}}{\max(|x|) + \epsilon}ight), -M, Might) \cdot rac{\max(|x|) + \epsilon}{M_{	ext{fp8}}}$$ <!-- source: Codes/framework/attentions/aegis_spectron_fp8.py lines 65-80 -->
2. **Spectral Routing**: Distributes tokens between local sliding window and global linear attention based on token singular value entropy.
3. **Factorized Linear Projections**: Projections execute as low-rank base factors plus trainable residual LoRA adapters. <!-- source: Codes/framework/attentions/aegis_spectron_fp8.py line 13 -->
