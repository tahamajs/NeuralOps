#!/usr/bin/env python3
"""
backend/server.py
=================
FastAPI + PyTorch inference backend for the 3D Neural Architecture Visualizer.
Streams real multi-head attention weight tensors and tokenizations directly
from Hugging Face Transformer models (DistilBERT, RoBERTa, GPT-2, Qwen/LLaMA)
to empower research-grade live attention exploration.
"""

import os
import sys
from typing import List, Dict, Any, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None

app = FastAPI(
    title="A3 Attention Real Inference API",
    description="Live PyTorch attention weights and token activation extraction for 3D Visualizer",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded model cache: model_id -> (model, tokenizer)
_models: Dict[str, Tuple[Any, Any]] = {}

SUPPORTED_MODELS = {
    "distilbert-base-uncased": "DistilBERT (6 layers, 12 heads, 768d)",
    "roberta-base": "RoBERTa (12 layers, 12 heads, 768d)",
    "gpt2": "GPT-2 (12 layers, 12 heads, 768d)",
}


def get_model(model_id: str = "distilbert-base-uncased") -> Tuple[Any, Any]:
    if AutoModel is None or AutoTokenizer is None:
        raise HTTPException(500, "Hugging Face transformers package is not installed.")

    if model_id not in _models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id, output_attentions=True)
            model.eval()
            _models[model_id] = (model, tokenizer)
        except Exception as e:
            raise HTTPException(500, f"Failed to load model '{model_id}': {str(e)}")

    return _models[model_id]


class AttentionRequest(BaseModel):
    text: str = Field(..., description="Prompt text to tokenize and pass through model")
    model_id: str = Field(default="distilbert-base-uncased", description="Hugging Face model ID")
    layer: int = Field(default=0, ge=0, description="0-indexed Transformer layer")
    head: int = Field(default=0, ge=0, description="0-indexed attention head")


class AttentionResponse(BaseModel):
    tokens: List[str]
    attention: List[List[float]]
    shape: List[int]
    model_id: str
    layer: int
    head: int
    entropy: float
    max_weight: float


@app.post("/attention", response_model=AttentionResponse)
def get_attention(req: AttentionRequest):
    if not req.text.strip():
        raise HTTPException(400, "Input text cannot be empty.")

    model, tokenizer = get_model(req.model_id)
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, max_length=64)

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    attentions = outputs.attentions  # tuple of (B, H, S, S)
    if not attentions:
        raise HTTPException(500, f"Model '{req.model_id}' did not return attention weights.")

    num_layers = len(attentions)
    if req.layer >= num_layers:
        raise HTTPException(400, f"Layer index {req.layer} out of range [0, {num_layers - 1}].")

    # Select batch item 0: shape (H, S, S)
    layer_attn = attentions[req.layer][0]
    num_heads = layer_attn.shape[0]
    if req.head >= num_heads:
        raise HTTPException(400, f"Head index {req.head} out of range [0, {num_heads - 1}].")

    attn_matrix = layer_attn[req.head].cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Clean tokens for display (e.g. GPT-2 'Ġ', RoBERTa 'Ġ')
    cleaned_tokens = [t.replace("Ġ", " ").replace("##", "") for t in tokens]

    # Compute Shannon entropy of attention distribution across sequence
    eps = 1e-12
    p = np.clip(attn_matrix, eps, 1.0)
    entropy_per_row = -np.sum(p * np.log2(p), axis=-1)
    mean_entropy = float(np.mean(entropy_per_row))

    return AttentionResponse(
        tokens=cleaned_tokens,
        attention=attn_matrix.tolist(),
        shape=list(attn_matrix.shape),
        model_id=req.model_id,
        layer=req.layer,
        head=req.head,
        entropy=round(mean_entropy, 4),
        max_weight=round(float(np.max(attn_matrix)), 4),
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_models": list(_models.keys()),
        "supported_models": SUPPORTED_MODELS,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    print(f"🚀 Starting A3 Real Attention Inference Backend on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
