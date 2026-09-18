# 9. Discussion

## 9.1 Key Takeaways

### 9.1.1 Takeaway 1: LoRA Reduces Training Cost

Trainable parameters reduced by **2.1\times×**. Training memory 
reduced by **2.1\times×**. This makes fine-tuning feasible on consumer 
GPUs.

### 9.1.2 Takeaway 2: Attention LoRA is Effective

For classification tasks, LoRA on attention outperforms LoRA on FFN. This 
suggests attention projections encode task-relevant transformations.

### 9.1.3 Takeaway 3: Merge Eliminates Inference Cost

After merging, the model has zero additional inference cost. This makes 
LoRA Attention strictly better than adapter methods that add latency.

### 9.1.4 Takeaway 4: Recovery for Compression

LoRA is an excellent recovery mechanism for compression methods. Our 
experiments show 90%+ recovery within 50 training steps.

## 9.2 When to Use LoRA Attention

### Recommended
- **Fine-tuning on consumer GPUs**
- **Multi-task adaptation** (train multiple LoRAs, swap at inference)
- **Post-compression recovery**
- **Few-shot fine-tuning**

### Not Recommended
- **Parameter-count-sensitive deployments** (before merge)
- **Tasks requiring major distribution shift** (full FT may be better)

## 9.3 Decision Flowchart

```
Need to fine-tune a large model? ─── Yes ─── Use LoRA Attention
  │ No
  ▼
Need to reduce model size? ─── Yes ─── Use A3/AASVD
  │ No
  ▼
Keep as is
```

## 9.4 Cost-Benefit Analysis

For DistilBERT fine-tuning:

| Metric | Full FT | LoRA Attention |
|---|---|---|
| Trainable params | 66.9M | 0.85 |
| Optimizer state | ~670 MB | ~0.85 MB |
| Training time (50 steps) | OOM | 1.42x s |
| Accuracy | 90.71% | 89.91% |
| Inference after merge | Baseline | Baseline |

## 9.5 Comparison with Published Methods

| Method | Architecture | Target | Mergeable |
|---|---|---|---|
| Adapters | Various | FFN | No |
| Prefix Tuning | Various | KV | No |
| **LoRA (standard)** | Various | FFN | Yes |
| **LoRA Attention** | **4** | **Attn** | **Yes** |

## 9.6 Deployment Considerations

### Pre-Merge

LoRA adapters add ~84.5 MB per model. This is small but 
non-zero.

### Post-Merge

Zero overhead. The model is identical to baseline.

### Multi-Task

Multiple LoRA adapters can be stored and swapped:

```python
# Load task A
model.load_lora("task_a.pt")
# Inference...

# Swap to task B
model.load_lora("task_b.pt")
# Inference...
```

Each adapter is only ~84.5 MB.

## 9.7 Open Questions

1. **Optimal rank per projection:** Is $r_L$ the same for Q, K, V, O?
2. **Rank allocation:** Can AdaLoRA-style allocation improve results?
3. **Cross-task transfer:** Does a LoRA trained on task A help task B?
4. **Interaction with compression:** Best order of compress-then-LoRA?

## 9.8 Broader Impact

### Positive
- Democratizes fine-tuning
- Reduces environmental cost
- Complements compression

### Negative
- Adds a training step (can't just download and use)
- Requires some ML expertise

---