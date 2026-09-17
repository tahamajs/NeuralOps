# 4. Method Formulation
1. **Factorised Projections**: Projections parameterized as $y = A(B x) + b$ with rank $r=128$.
2. **Local Context**: Window $W=128$ for high-fidelity syntactic attention.
3. **Global Context**: Linear kernel approximation $\phi(x) = \operatorname{elu}(x) + 1$.
4. **LoRA Recovery**: Trainable low-rank adapters fine-tune the compressed backbone. <!-- source: Codes/framework/attentions/aegis_v2.py line 50 -->
