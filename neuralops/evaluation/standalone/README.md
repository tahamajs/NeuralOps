# NeuralOps evaluation experiments

These programs are optional, explicit research runs—not part of package import
or the default benchmark suite. They are launched in separate processes so
model and accelerator state is released when each run exits:

```bash
pip install -e ".[cuda,evaluation-experiments]"
neuralops-eval list
neuralops-eval run comprehensive --model-path gpt2 --output-dir results/evaluation
neuralops-eval run comprehensive --context-lengths 128 512 1024 --device cuda:0
neuralops-eval run ultra-long-context --model-path gpt2 --output-dir results/evaluation
```

The model path can be a Hugging Face model ID or a local directory. The
launcher creates the output directory and exports `NEURALOPS_MODEL_PATH` and
`NEURALOPS_OUTPUT_DIR` for the selected run. Most experiments require CUDA;
the comprehensive matrix accepts CPU execution, although its reported
performance metrics are not comparable to GPU timings. Use
`CUDA_VISIBLE_DEVICES` to select a GPU.

## Interpretation and limitations

- Reasoning prompts in these scripts are synthetic stress inputs. In
  particular, `arithmetic-continuation` is not the GSM8K dataset and does not
  measure benchmark accuracy. Its target check is an exact-number diagnostic,
  not a standardized reasoning-quality metric.
- Autoregressive loops repeatedly evaluate the complete growing prefix. They
  do not measure an incremental KV-cache decode path, so memory growth from
  these loops is not evidence for or against constant-size recurrent state.
- Long-context inputs are synthetic (random token IDs or repeated calibration
  text), not a representative language-modeling corpus. Perplexity and
  scaling results must be labeled with that protocol.
- The concurrent offloading experiment adds synthetic background GPU work; it
  does not implement or benchmark production KV-cache offloading.
- The comprehensive matrix records errors per configuration. A generated
  report is not automatically a correctness certification or a release gate.

These experiments are retained for targeted engineering investigation. For
canonical, reproducible benchmark workflows, use `python -m
neuralops.benchmarks` and its documented result ledger.
