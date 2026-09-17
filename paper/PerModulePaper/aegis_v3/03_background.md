# 3. Mathematical Background
For local output $Y_{	ext{local}}$ and global output $Y_{	ext{global}}$, dynamic gating computes $g = \sigma(X W_g + b_g)$ and outputs $Y = g \odot Y_{	ext{local}} + (1 - g) \odot Y_{	ext{global}}$.
