# Universal Benchmarking Guide: Tasks & Architectures

The NeuralOps Universal Benchmarking Engine allows evaluating and fine-tuning any of the **232 registered attention and compressor modules** across **14 distinct task formats** and **arbitrary HuggingFace or custom model architectures**.

---

## 1. Supported Task Types

| Task Identifier | HF Pipeline / AutoModel | Default Metrics | Example Datasets |
|:---|:---|:---|:---|
| `classification` | `AutoModelForSequenceClassification` | Accuracy, F1, Precision, Recall | `sst2`, `mrpc`, `imdb`, CSV |
| `question_answering` | `AutoModelForQuestionAnswering` | Exact Match (EM), F1 | `squad`, `squad_v2` |
| `summarization` | `AutoModelForSeq2SeqLM` | ROUGE-1, ROUGE-2, ROUGE-L | `cnn_dailymail`, `xsum` |
| `translation` | `AutoModelForSeq2SeqLM` | BLEU, SacreBLEU | `wmt16`, `opus_books` |
| `instruction_tuning` | `AutoModelForCausalLM` | Token Perplexity, Loss | `tatsu-lab/alpaca` |
| `language_modeling` | `AutoModelForCausalLM` | Perplexity, BPC | `wikitext2`, `c4`, `ptb` |
| `regression` | `AutoModelForSequenceClassification` | MSE, RMSE, MAE, Pearson, Spearman | `stsb`, local CSV |
| `multiple_choice` | `AutoModelForMultipleChoice` | Accuracy, Log Likelihood | `arc`, `hellaswag`, `mmlu` |

---

## 2. Universal CLI Options

Run benchmarks directly with explicit task specification or automatic task inference:

```bash
# Evaluate a custom local CSV dataset on sequence classification
neuralops run --module svd_linear_attention --model bert-base-uncased --dataset my_data.csv --task classification --text-column sentence --label-column target

# Evaluate summarization on CNN/DailyMail
neuralops run --module dense_attention --model t5-small --dataset cnn_dailymail --task summarization --max-samples 500

# Evaluate instruction tuning on Alpaca with LLaMA
neuralops run --module fast_attention --model meta-llama/Llama-2-7b-hf --dataset tatsu-lab/alpaca --task instruction_tuning --hf-token $HF_TOKEN
```

---

## 3. Results Management & Persistence

Every execution automatically records:
1. Canonical **15-section JSON schema** (`results/<run_id>.json`)
2. Append-only **JSONL** and **CSV** logs
3. Formatted **Markdown run summaries**
4. Compressed **Parquet** analytical records
5. Exact reproduction commands and standalone reproduction scripts
