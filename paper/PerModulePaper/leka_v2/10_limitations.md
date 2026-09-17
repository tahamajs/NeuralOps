# 10. Limitations

## 10.1 Not Truly Linear at Default ρ

The landmark attention is $O(L \cdot L_\text{max})$, where $L_\text{max} = \rho L$. 
At $\rho = 0.5$, this is still $O(L^2/2)$.

## 10.2 Parameter Increase

LEKAv2 adds ~66.4M% parameters per layer.

## 10.3 Fatal Error Handling

`copy_weights` raises `RuntimeError` on extractor failure.

## 10.4 Cross-Attention

Not supported.

## 10.5 Task Coverage

Only SST-2.

## 10.6 Model Coverage

| Size | Tested |
|---|---|
| ≤124M | Yes |
| 1B+ | No |

## 10.7 Hardware

Only RTX 4070.

## 10.8 No Multi-Seed

Single seed.

## 10.9 Landmark Scorer Complexity

The scorer is a small MLP; its accuracy depends on training data.

## 10.10 Local Gate Saturation

If $g_h$ saturates, one branch is ignored.

## 10.11 Correction Matrix Scaling

$W_\text{corr}$ is $O(H d_h^2)$ — significant for large $d_h$.

## 10.12 Code Quality

Research-grade.

## 10.13 Ethical Considerations

See Appendix E.

---