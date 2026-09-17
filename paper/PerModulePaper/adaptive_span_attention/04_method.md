# 4. Method Formulation

## 4.1 Learnable Dynamic Span
Each head $h \in \{1, \dots, H\}$ possesses a learnable scalar $\text{logit}_h$:
$$z_h = \sigma(\text{logit}_h) \cdot S_{\max}$$ <!-- source: Codes/framework/attentions/adaptive_span_attention.py line 9 -->
where $S_{\max}$ is the maximum span (default 128) and $\sigma$ is the sigmoid function.

## 4.2 Soft Masking Function
With ramp size $R$ (default 32.0), the soft ramp function is defined as:
$$g_{z_h}(d) = \operatorname{clamp}\left(\frac{R + z_h - d}{R}, 0, 1\right)$$ <!-- source: Codes/framework/attentions/adaptive_span_attention.py line 10 -->
For $d \le z_h - R$, $g = 1$; for $d \ge z_h$, $g = 0$; within $[z_h - R, z_h]$, it decays linearly from 1 to 0.

## 4.3 Output Aggregation
$$A_{ij}^{(h)} = \operatorname{Softmax}\left(S_{ij}^{(h)} + M_{\text{causal}} + M_{\text{pad}}\right) \cdot g_{z_h}(d_{ij})$$ <!-- source: Codes/framework/attentions/adaptive_span_attention.py line 13 -->
$$O^{(h)} = A^{(h)} V^{(h)}$$ <!-- source: Codes/framework/attentions/adaptive_span_attention.py line 14 -->
