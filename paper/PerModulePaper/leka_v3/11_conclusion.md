# 11. Conclusion

## 11.1 Summary

We presented **LEKAv3**, a linear-time attention mechanism combining 
learned landmark selection with a local sliding window. LEKAv3 achieves 
$\mathcal{O}(L L_\text{land} + L W)$ complexity, scales to long sequences, 
and provides a learnable alternative to fixed landmark selection.

## 11.2 Results Recap

| Model | Acc Δ | Landmark | Window |
|---|---|---|---|
| DistilBERT | {{D_DELTA}}% | {{L0}} | {{WINDOW}} |
| RoBERTa | {{R_DELTA}}% | {{L0}} | {{WINDOW}} |
| GPT-2 | {{G_DELTA}}% | {{L0}} | {{WINDOW}} |

## 11.3 Key Takeaways

1. **Linear complexity is real.**
2. **Learned landmarks beat fixed selection.**
3. **Local window preserves detail.**
4. **Gated fusion learns per-head balance.**
5. **Parameter count increases** — honest framing.

## 11.4 Future Work

### Short-Term
- Fused Triton kernel
- Adaptive $L_\text{land}$ per layer
- INT8 combined

### Long-Term
- Theoretical bound on approximation error
- Cross-task scorer transfer
- Multimodal extension

## 11.5 Closing Remarks

LEKAv3 is a practical linear-time attention for long-context models. Code 
and checkpoints released.

## 11.6 Acknowledgments

{{ACKNOWLEDGMENTS}}

## 11.7 Author Contributions

- **{{AUTHOR_1}}** designed LEKAv3, implemented, ran experiments, wrote paper.
- **{{AUTHOR_2}}** provided feedback, edited manuscript.

## 11.8 Conflict of Interest

None.
```

---