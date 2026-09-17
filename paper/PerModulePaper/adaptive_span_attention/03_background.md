# 3. Mathematical Background
Standard scaled dot-product attention computes:
$$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d_h}}$$
For autoregressive models, relative distance is $d = i - j \ge 0$. For bidirectional encoders, relative distance is $d = |i - j|$.
