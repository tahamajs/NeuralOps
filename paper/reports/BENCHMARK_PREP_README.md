# Benchmark Preparation Status

## Files Prepared

### Decoder Evaluation
- [`Codes/framework/finetune_gpt2_wikitext.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/finetune_gpt2_wikitext.py) (READY)
- [`Codes/framework/run_gpt2_sweep.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_gpt2_sweep.py) (READY)
- Modified: [`Codes/framework/run_one_module_isolated.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_one_module_isolated.py) (`--perplexity` flag added)

### Long-Context Evaluation
- [`Codes/framework/data_utils/load_byte_imdb.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/data_utils/load_byte_imdb.py) (READY)
- [`Codes/framework/run_long_context_sweep.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_long_context_sweep.py) (READY)
- Modified: [`Codes/framework/run_one_module_isolated.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_one_module_isolated.py) (`--long_context` flag added)

### Pareto Metrics
- Modified: [`Codes/framework/run_one_module_isolated.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_one_module_isolated.py) (VRAM / latency / throughput telemetry)
- Modified: [`Codes/framework/run_quality_sweep.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/run_quality_sweep.py) CSV schema (5 new columns: `latency_ms_per_token`, `throughput_tok_per_sec`, `perplexity`, `long_context_accuracy`)

### Plotting
- [`Codes/framework/plot_pareto.py`](file:///Users/tahamajs/Documents/uni/Project/Codes/framework/plot_pareto.py) (READY)

### LaTeX Tables
- [`paper/iclr2027-submission/tables/table6_decoder.tex`](file:///Users/tahamajs/Documents/uni/Project/paper/iclr2027-submission/tables/table6_decoder.tex) (TEMPLATE)
- [`paper/iclr2027-submission/tables/table7_long_context.tex`](file:///Users/tahamajs/Documents/uni/Project/paper/iclr2027-submission/tables/table7_long_context.tex) (TEMPLATE)
- [`paper/iclr2027-submission/tables/table8_pareto.tex`](file:///Users/tahamajs/Documents/uni/Project/paper/iclr2027-submission/tables/table8_pareto.tex) (TEMPLATE)

## What to Review

1. **Do the scripts look correct?**
   - Clean, modular CLI structures with standard argparse interfaces and typing annotations.
2. **Are the argparse flags correct?**
   - Defaults preserve backward compatibility without affecting existing encoder sweeps.
3. **Is the perplexity calculation correct?**
   - Implements standard sliding-window evaluation with `max_length=1024`, `stride=512`, masking overlap tokens with `-100`, and calculating `exp(sum(NLL) / total_target_tokens)`.
4. **Is the long-context data loader correct?**
   - Enforces fixed 4096 sequence length, batch size 2, with fallback from HuggingFace hub to raw UTF-8 byte conversion.
5. **Are the LaTeX table structures correct?**
   - Booktabs formatting with directional arrows ($\uparrow$, $\downarrow$) matching standard ICLR publication styling.

## Not Yet Executed

All scripts are fully prepared and syntax-checked, but **NOT** run. Awaiting user approval before executing any sweep.

## Estimated Run Times (When Approved)

- **GPT-2 Fine-Tuning**: ~1 hour on RTX 4070
- **GPT-2 Sweep (12 modules)**: ~30 minutes
- **Long-Context Sweep (12 modules, 4096 tokens)**: ~2 hours
- **Pareto Metrics on Existing Runs**: 0 extra GPU runs needed (offline plotting & post-processing)
