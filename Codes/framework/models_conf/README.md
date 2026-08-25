# Models Configuration & Variants Subsystem (`Codes/framework/models_conf`)

The `models_conf` subsystem provides foundation model loaders, Parameter-Efficient Fine-Tuning (PEFT) adapters, multi-backend quantization engines, and model export pipelines.

---

## 🏗 Architecture Overview

- **Model Loaders (`loaders/`, `core_loaders.py`)**: Factory orchestrator (`ModelOrchestrator`) for loading pretrained transformer models from HuggingFace Hub, Safetensors, FlashAttention, and Unsloth.
- **PEFT Variants (`variants/peft/`)**: Modular PEFT adapter injection registered in `PEFT_REGISTRY`:
  - **LoRA & QLoRA (`lora.py`, `qlora.py`)**: Low-Rank Adaptation with 4-bit/8-bit base weight quantization.
  - **IA3, Prefix & Prompt Tuning (`ia3.py`, `prefix_tuning.py`, `prompt_tuning.py`)**: Key-Value activation scaling and virtual prompt token embeddings.
- **Quantization Backends (`variants/quantization/`)**: Integrated backends:
  - **BitsAndBytes (`bitsandbytes.py`)**: 4-bit (NF4/FP4) and 8-bit quantization.
  - **AWQ & GPTQ (`awq.py`, `gptq.py`)**: Activation-aware Weight Quantization and second-order Taylor error minimization.
  - **TorchAO & FP8 (`torchao.py`, `fp8.py`)**: PyTorch native FP8 and int4/int8 execution.
- **Export Pipelines (`variants/export/`)**: Deployment exporters:
  - **ONNX (`onnx.py`)**, **TorchScript (`torchscript.py`)**, **TensorRT (`tensorrt.py`)**, and **TFLite (`tflite.py`)**.

---

## 📦 Key Files & Modules

| Module | Description |
|:---|:---|
| `core_loaders.py` | Base loader functions and model configuration utilities. |
| `loaders/orchestrator.py` | `ModelOrchestrator` managing loading, quantization, and adapter injection. |
| `variants/peft/` | PEFT adapters (LoRA, QLoRA, IA3, Prefix Tuning, Prompt Tuning). |
| `variants/quantization/` | Quantization engines (BitsAndBytes, AWQ, GPTQ, TorchAO, FP8). |
| `variants/export/` | Model exporters for ONNX, TorchScript, TensorRT, and TFLite runtimes. |

---

## 🚀 Quick Start & Usage

### 1. Loading Model with LoRA & Quantization
```python
from framework.models_conf.loaders.orchestrator import ModelOrchestrator

orchestrator = ModelOrchestrator()
model = orchestrator.load_model(
    model_name="distilbert-base-uncased",
    peft_method="lora",
    peft_kwargs={"r": 16, "lora_alpha": 32},
)
```

### 2. Exporting Model to ONNX or TFLite
```python
from framework.models_conf.variants.export.onnx import ONNXExporter

exporter = ONNXExporter()
exporter.export(model, output_path="./exported/model.onnx")
```
