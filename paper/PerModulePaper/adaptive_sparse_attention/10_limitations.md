# 10. Limitations
On bidirectional encoders with complex syntax (e.g. RoBERTa), aggressive sparsity can degrade accuracy (dropping from 78.56% to 64.79%) unless conservative sparsity ratios ($s \le 0.1$) are enforced.
