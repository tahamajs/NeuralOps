Module neuralops.benchmarks.run_ruler_benchmark
===============================================
Needle-in-a-Haystack (NIAH) Context Retrieval Benchmark (RULER-style) for GPT-2.

Evaluates long-context retrieval capabilities of attention and compression modules
across multiple context lengths (e.g. 512, 1024, 2048, 4096) and depths (0.1, 0.25, 0.5, 0.75, 0.9).

Features:
  1. Diverse fixed needle triples (code, city, date, element, password, etc.).
  2. Multi-depth placement across synthetic neutral haystack tokens.
  3. Automatic context clamping to model positional limits (default 1024 for standard GPT-2).
  4. Resilient OOM and exception handling (skips/records failed context lengths cleanly).
  5. JSON and Markdown summary table output.

Functions
---------

`construct_niah_prompt(tokenizer: Any, needle: str, question: str, depth: float, effective_length: int) ‑> list[int]`
:   Construct token IDs for needle inserted into haystack filler at the requested depth ratio.
    The final tokens will end with question and answer prompt.

`evaluate_context_length(model: torch.nn.Module, tokenizer: Any, context_len: int, depths: list[float], num_samples: int, max_position: int, device: torch.device) ‑> dict[str, Any]`
:   Evaluate retrieval accuracy across num_samples for a given context length.
    Clamps effective prompt length to min(context_len, max_position).

`main(argv: list[str] | None = None) ‑> int`
:   

`parse_args(argv: list[str] | None = None) ‑> argparse.Namespace`
:   

`print_summary_table(results: dict[str, Any], depths: list[float], module_name: str) ‑> None`
:   Print summary table formatted like: ctx_len | accuracy | depth_0.1 | depth_0.5 | depth_0.9