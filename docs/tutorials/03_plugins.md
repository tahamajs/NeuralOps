# Tutorial 3: Serving Adapters & Production Deployment

NeuralOps serving adapters provide multi-framework interfaces for high-throughput concurrent inference.

## vLLM In-Tree PagedStateTable
```python
from neuralops.models.paged_state import PagedStateTable

table = PagedStateTable(num_pages=1024, num_heads=12, head_dim=64, device="cuda")
p0 = table.register_sequence(seq_id=1, initial_pages=1)
# Zero-copy sequence fork (e.g. beam search or parallel sampling)
p_fork = table.fork_sequence(parent_seq_id=1, child_seq_id=2)
# Automatic Copy-on-Write isolation upon write
table.prepare_write(seq_id=2, page_slot=0)
```

## SGLang Radix Tree Prefix Cache
```python
from neuralops.integrations.sglang import SGLangRadixAdapter

adapter = SGLangRadixAdapter(model=None, max_cached_tokens=4096)
# Match and reuse common system prompt prefixes across queries
matched_len, kv = adapter.match_prefix([101, 2054, 2003])
```
