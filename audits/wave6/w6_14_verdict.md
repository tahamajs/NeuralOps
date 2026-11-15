# Deletion Verdict & Rationale: Example 20

- **Target File:** `examples/20_long_context_niah_eval.py`
- **Verdict:** DELETED (`git rm examples/20_long_context_niah_eval.py`)
- **Agent:** W6-14
- **Date:** 2026-11-15

## Rationale
`examples/20_long_context_niah_eval.py` was thoroughly audited and found to violate the core tenets of Wave 6 (Law 1: No fake data for real evaluations, Law 2: No hardcoded pass status):
1. **Fabricated Attention Maps:** Lines 74 and 107 used `torch.randn` to synthesize random attention logit vectors across context lengths `[128, 256, 512, 1024, 32768]` and depth offsets, adding an artificial boost (+3.5 / +4.0) to simulate needle retrieval.
2. **Missing Real LM & Tokenizer:** There was no actual language model evaluation, tokenizer passkey insertion, haystack textual context, or retrieval extraction query.
3. **Simulated KV Eviction:** H2O (Heavy Hitter Oracle) KV cache eviction was simulated using `torch.topk` over random numbers rather than genuine transformer cache state.
4. **Hardcoded Pass Status:** Line 207 invoked `pass_fail("Overall Example 20 Status", True, ...)` with a literal boolean `True`.

## True NIAH Requirements
Authentic long-context Needle In A Haystack evaluation cannot run honestly without:
- An authentic causal language model (e.g., `transformers.AutoModelForCausalLM` or fine-tuned long-context model).
- Real text tokens spanning at least 8k-32k tokens.
- Insertion of a secret passkey needle into haystack paragraphs and verifying generative retrieval / token log-likelihood.
Such long-context evaluations require multi-GPU VRAM or dedicated evaluation suites and are not suitable for lightweight unit/examples CPU workflows.

## Canonical Replacements
For genuine NIAH and passkey retrieval benchmarking within NeuralOps:
- Refer to `neuralops/evaluation/niah.py` and `neuralops/evaluation/niah_passkey.py`.
- Benchmark suite: `neuralops/benchmarks/suites/niah.py` and YAML config `configs/niah.yaml`.
- Tests: `tests/evaluation/test_niah_passkey.py` and `tests/benchmarks/test_niah_real.py`.
