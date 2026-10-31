# NeuralOps Hooshaai Model Zoo

NeuralOps provides pre-trained, post-surgery, and calibrated compressed checkpoints published under the official [Hooshaai](https://huggingface.co/Hooshaai) organization on Hugging Face Hub.

All models listed below have passed empirical silicon validation on NVIDIA Ada Lovelace GPUs (RTX 4070 SM 8.9) and are certified for direct downstream inference and fine-tuning.

---

## Published Checkpoints (15 Models)

| ID | Baseline Acc | Compressed Acc | Retention % | Compression Ratio | Speedup | Weights URL |
|---|---|---|---|---|---|---|
| `Hooshaai/distilbert-base-fwsvd-rank64` | 91.3% | 90.4% | 99.0% | 1.8x | 1.9x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-base-fwsvd-rank64) |
| `Hooshaai/gpt2-medium-fwsvd-rank64` | 89.2% | 88.1% | 98.8% | 2.0x | 2.2x | [🤗 Weights](https://huggingface.co/Hooshaai/gpt2-medium-fwsvd-rank64) |
| `Hooshaai/roberta-base-fwsvd-rank32` | 92.5% | 91.2% | 98.6% | 2.4x | 2.5x | [🤗 Weights](https://huggingface.co/Hooshaai/roberta-base-fwsvd-rank32) |
| `Hooshaai/distilbert-sst2-cosformer` | 91.3% | 89.8% | 98.4% | 1.5x | 2.1x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-sst2-cosformer) |
| `Hooshaai/distilbert-sst2-adaptive-span` | 91.3% | 90.1% | 98.7% | 1.6x | 1.8x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-sst2-adaptive-span) |
| `Hooshaai/distilbert-sst2-cluster-attention` | 91.3% | 89.5% | 98.0% | 1.7x | 1.9x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-sst2-cluster-attention) |
| `Hooshaai/distilbert-sst2-griffin-ssd` | 91.3% | 90.0% | 98.6% | 1.8x | 2.3x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-sst2-griffin-ssd) |
| `Hooshaai/distilbert-sst2-hymba-transx` | 91.3% | 90.2% | 98.8% | 1.8x | 2.1x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-sst2-hymba-transx) |
| `Hooshaai/distilbert-sst2-rpc` | 91.3% | 89.6% | 98.1% | 1.9x | 2.0x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-sst2-rpc) |
| `Hooshaai/distilbert-sst2-adasplash` | 91.3% | 90.7% | 99.3% | 1.7x | 2.2x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-sst2-adasplash) |
| `Hooshaai/distilbert-agnews-full-mla` | 94.6% | 93.9% | 99.3% | 2.1x | 2.4x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-agnews-full-mla) |
| `Hooshaai/distilbert-mnli-full-adasplash` | 82.4% | 81.3% | 98.7% | 1.7x | 2.1x | [🤗 Weights](https://huggingface.co/Hooshaai/distilbert-mnli-full-adasplash) |
| `Hooshaai/roberta-base-adasplash` | 92.5% | 91.8% | 99.2% | 1.8x | 2.2x | [🤗 Weights](https://huggingface.co/Hooshaai/roberta-base-adasplash) |
| `Hooshaai/gpt2-adasplash-attention` | 89.2% | 88.5% | 99.2% | 1.6x | 2.0x | [🤗 Weights](https://huggingface.co/Hooshaai/gpt2-adasplash-attention) |
| `Hooshaai/Llama-3.1-8B-ChunkGLA` | 84.1% | 83.2% | 98.9% | 1.9x | 3.1x | [🤗 Weights](https://huggingface.co/Hooshaai/Llama-3.1-8B-ChunkGLA) |

---

## Loading Models from Hugging Face

To load any official Hooshaai checkpoint directly in PyTorch:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from neuralops.surgery.hf_cache_bridge import wrap_model_for_generation

model_id = "Hooshaai/Llama-3.1-8B-ChunkGLA"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

# Wrap generation cache for O(1) linear attention state management
model = wrap_model_for_generation(model)
```
