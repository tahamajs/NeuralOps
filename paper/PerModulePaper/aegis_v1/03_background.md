# 3. Mathematical Background
Local sliding window attention restricts interactions to relative distance $|i - j| \le W$ ($W=128$). Global linear attention uses positive feature mapping $\phi(x) = \operatorname{elu}(x) + 1$ to compute associative key-value accumulation $O = Q (\sum K^	op V) / (Q \sum K^	op)$.
