# Nested Encoder Check — phantom

## Purpose

Detect whether the model's attention is "flat" (BERT/DistilBERT) or "nested"
(RoBERTa/BERT). Setting `is_nested_encoder` wrong destroys the output
projection.

## Detection

```python
def detect_nested_encoder(attn_module):
    """Return True if the module is a nested encoder container."""
    # RoBERTa: attention.self + attention.output.dense
    if hasattr(attn_module, "self") and hasattr(attn_module, "output"):
        if hasattr(attn_module.output, "dense"):
            return True
    # DistilBERT: flat, has out_lin inside attention
    if hasattr(attn_module, "out_lin"):
        return False
    # BERT: same as RoBERTa
    if hasattr(attn_module, "self") and hasattr(attn_module.self, "query"):
        return True
    return False
```

## Expected values

| Model | `is_nested_encoder` | `out_proj` applied internally? |
|---|---|---|
| DistilBERT | `False` | Yes |
| RoBERTa | `True` | No (outer `output.dense` does it) |
| BERT | `True` | No |
| GPT-2 | `False` | Yes |
| Qwen/LLaMA | `False` | Yes |

## Verification

```python
# In forward, when is_nested_encoder=True, context should NOT be out_proj'd.
# Test: run forward, check output.shape == hidden.shape
```

## Output

Save to `papers/modules/phantom/metrics/nested_encoder_audit.json`:

```json
{
  "distilbert": {"is_nested": false, "detected_correctly": true},
  "roberta": {"is_nested": true, "detected_correctly": true},
  "gpt2": {"is_nested": false, "detected_correctly": true}
}
```
