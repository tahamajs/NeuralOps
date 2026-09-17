# 3. Mathematical Background
FP8 E4M3 uses 1 sign, 4 exponent, and 3 mantissa bits (dynamic range $pprox 448$). Dynamic scaling $x_{	ext{scaled}} = x \cdot rac{	ext{max\_val}}{\max(|x|) + \epsilon}$ matches tensor values to available dynamic range. <!-- source: Codes/framework/attentions/aegis_spectron_fp8.py lines 43, 65-70 -->
