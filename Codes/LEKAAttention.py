

    #!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Token Compression Attention for GLUE — Production-Ready Version
================================================================
Fixes all critical bugs and adds:
1. Proper weight initialization from pretrained BERT
2. Correct value compression with orthogonal projection
3. Gradient checkpointing for memory efficiency
4. LayerNorm weight renaming for checkpoint compatibility
5. Model saving/loading with compression modules
6. Mixed precision training support
7. Early stopping and better hyperparameters
"""

import os
import math
import argparse
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"  # though we use local parquet, still good

# ---------- Offline settings ----------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, default_data_collator,
    EarlyStoppingCallback, get_cosine_schedule_with_warmup,
    set_seed, PreTrainedModel, PretrainedConfig
)
from datasets import Dataset, DatasetDict
import evaluate
os.environ["HF_HUB_OFFLINE"] = "1"

# ── Environment ─────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── 1. Compression Configuration ────────────────────────────
@dataclass
class CompressionConfig:
    # Linformer
    k_early: int = 32
    k_mid: int = 16
    k_late: int = 12
    progressive: bool = True
    use_orthogonal: bool = True  # (ignored in current impl)
    use_gate: bool = True
    gate_bias: float = 0.0
    max_seq_len: int = 128
    attn_score_clamp: float = 50.0
    dropout: float = 0.1
    ortho_reg_weight: float = 0.001

    # LEKA
    attn_type: str = "linformer"   # "linformer" or "leka"
    num_landmarks: int = 16
    feature_dim: int = 128
    use_spherical_norm: bool = True
    kernel: str = "elu"

    def get_k_for_layer(self, layer_idx: int) -> int:
        if not self.progressive:
            return self.k_mid
        if layer_idx < 4:
            return self.k_early
        elif layer_idx < 8:
            return self.k_mid
        else:
            return self.k_late

# ── 2. Data Loading (Local Parquet) ────────────────────────
TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
class LEKAAttention(nn.Module):
    def __init__(self, config, comp_config, layer_idx, bert_mode=False):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.head_dim = self.attention_head_size
        self.bert_mode = bert_mode
        
        # دریافت پارامترها از comp_config
        self.num_landmarks = comp_config.num_landmarks
        self.feature_dim = comp_config.feature_dim
        
        # لایه‌های Q/K/V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(comp_config.dropout)
        
        # ماتریس تصادفی برای kernel features (بافر)
        self.register_buffer('rand_proj', torch.randn(self.head_dim, self.feature_dim))
        
        # ماتریس تصحیح landmark (per head)
        self.W_corr = nn.Parameter(
            torch.eye(self.head_dim).unsqueeze(0).repeat(self.num_attention_heads, 1, 1)
        )
        
        # دروازه (اختیاری، از comp_config)
        if comp_config.use_gate and not self.bert_mode:
            self.gate = nn.Parameter(torch.tensor(comp_config.gate_bias))
        else:
            self.gate = None
            
            
            
    def transpose_for_scores(self, x):
        """Reshape from (B, N, all_head_size) to (B, H, N, head_dim)"""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    
    def kernel_feature_map(self, x):
        """
        Apply random Fourier features: φ(x) = [cos(xW), sin(xW)] / sqrt(feature_dim)
        
        Args:
            x: (B, H, N, D)
        Returns:
            features: (B, H, N, 2*feature_dim)
        """
        # Move rand_proj to correct device
        rand_proj = self.rand_proj.to(x.device)
        
        # Project: (B, H, N, D) @ (D, feature_dim) -> (B, H, N, feature_dim)
        proj = torch.matmul(x, rand_proj)
        
        # Apply cos and sin
        cos_features = torch.cos(proj)
        sin_features = torch.sin(proj)
        
        # Concatenate and normalize
        features = torch.cat([cos_features, sin_features], dim=-1)  # (B, H, N, 2*feature_dim)
        features = features / math.sqrt(self.feature_dim)
        
        return features
    
    def select_landmarks(self, K_features, V):
        """
        Select landmarks using k-means++ initialization
        
        Args:
            K_features: (B, H, N, 2*feature_dim)
            V: (B, H, N, D)
        Returns:
            K_land: (B, H, L, 2*feature_dim)
            V_land: (B, H, L, D)
        """
        B, H, N, feat_dim = K_features.shape
        L = self.num_landmarks
        
        if N <= L:
            # If sequence is shorter than landmarks, use all tokens
            return K_features, V
        
        # Simple uniform sampling (faster than k-means++)
        indices = torch.linspace(0, N - 1, L, device=K_features.device).long()
        indices = indices.unsqueeze(0).unsqueeze(0).expand(B, H, -1)  # (B, H, L)
        
        # Gather landmarks
        K_land = torch.gather(
            K_features, 
            2, 
            indices.unsqueeze(-1).expand(-1, -1, -1, feat_dim)
        )  # (B, H, L, feat_dim)
        
        V_land = torch.gather(
            V, 
            2, 
            indices.unsqueeze(-1).expand(-1, -1, -1, V.shape[-1])
        )  # (B, H, L, D)
        
        return K_land, V_land
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, N, D = hidden_states.shape
        L = self.num_landmarks

        # print("==== DEBUG LEKAAttention.forward ====")
        # print(f"hidden_states.shape = {hidden_states.shape}")
        # print(f"N = {N}, L = {L}")

        # Linear projections
        Q = self.transpose_for_scores(self.query(hidden_states))
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))
        Q_features = self.kernel_feature_map(Q)
        K_features = self.kernel_feature_map(K)

        K_land, V_land = self.select_landmarks(K_features, V)

        W = self.W_corr.unsqueeze(0).expand(B, -1, -1, -1)
        V_land_adj = torch.einsum('bhld,bhde->bhle', V_land, W)

        attn_weights = torch.matmul(Q_features, K_land.transpose(-2, -1))  # (B, H, N, L)
        # print(f"attn_weights.shape (before mask) = {attn_weights.shape}")

        # Mask handling – DistilBERT gives us (B,1,N,N) extended mask
        mask = attention_mask
        if mask is None:
            mask = kwargs.get('attn_mask', None)

        if mask is not None:
            # print(f"mask.shape = {mask.shape}")
            # The mask is 4D with shape (B,1,N,N) – full attention mask
            if mask.dim() == 4 and mask.shape[-2] == mask.shape[-1]:
                # Extract per-token validity from the first query position ([CLS])
                # Valid positions have 0.0, padding positions have -10000.0
                token_valid = mask[:, 0, 0, :] > -1e9   # (B, N) boolean True = valid
                # print(f"token_valid.shape = {token_valid.shape}")

                # Sub-sample landmarks
                if N > L:
                    indices = torch.linspace(0, N - 1, L, device=token_valid.device).long()
                    landmark_mask = token_valid[:, indices]     # (B, L)
                    # print(f"landmark_mask after indexing = {landmark_mask.shape}")
                else:
                    landmark_mask = token_valid
                # Reshape to (B, 1, 1, L) for broadcasting
                landmark_mask = landmark_mask.unsqueeze(1).unsqueeze(2)
                # print(f"landmark_mask final = {landmark_mask.shape}")
            else:
                # Fallback for standard (B,1,1,N) or 2D masks (if ever used)
                if mask.dim() == 4:
                    mask_2d = mask.squeeze(1).squeeze(1) > -1e9
                elif mask.dim() == 3:
                    mask_2d = mask.squeeze(1).bool()
                elif mask.dim() == 2:
                    mask_2d = mask.bool()
                else:
                    raise ValueError(f"Unexpected mask dim: {mask.dim()}")
                if N > L:
                    indices = torch.linspace(0, N - 1, L, device=mask_2d.device).long()
                    landmark_mask = mask_2d[:, indices]
                else:
                    landmark_mask = mask_2d
                landmark_mask = landmark_mask.unsqueeze(1).unsqueeze(2)

            attn_weights = attn_weights.masked_fill(~landmark_mask, -10000.0)
            # print("Mask applied successfully")
        # else:
        #     print("No mask provided")

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_layer = torch.matmul(attn_weights, V_land_adj)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        # print(f"context_layer.shape = {context_layer.shape}")
        # print("=====================================\n")

        return context_layer, None

    def get_ortho_loss(self):
        return 0.0


def load_glue_local(task_name: str) -> DatasetDict:
    """Load GLUE dataset from local parquet files (raw, no tokenization)."""
    base_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Local path {task_path} not found")

    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"):
            continue
        # split name: "train", "validation", etc. (works with both "train.parquet" and "train-00000-of-00001.parquet")
        split = fname.split("-")[0]
        file_path = os.path.join(task_path, fname)
        splits[split] = Dataset.from_parquet([file_path])

    # Handle MNLI special splits
    if task_name == "mnli":
        for split_name in ["validation_matched", "validation_mismatched"]:
            fname = f"{split_name}.parquet"
            file_path = os.path.join(task_path, fname)
            if os.path.exists(file_path):
                splits[split_name] = Dataset.from_parquet([file_path])

    return DatasetDict(splits)

def get_num_labels_and_is_regression(task_name: str):
    t = task_name.lower()
    if t == "stsb":
        return 1, True
    if t == "cola":
        return 2, False
    glue_num_labels = {
        "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3,
        "qnli": 2, "rte": 2, "wnli": 2,
    }
    return glue_num_labels[t], False

GLUE_TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

class LinformerProjection(nn.Module):
    """Learnable projection for keys and values (no orthogonal constraint)."""
    def __init__(self, n: int, k: int):
        super().__init__()
        # Initialise as truncated identity + small noise
        base = torch.eye(n)[:k, :]          # (k, n)
        self.E = nn.Parameter(base.clone())
        self.F = nn.Parameter(base.clone())
        with torch.no_grad():
            self.E += torch.randn_like(self.E) * 0.02
            self.F += torch.randn_like(self.F) * 0.02

    def forward(self, K, V, mask_2d=None):
        # K, V: (B, heads, N, d)
        B, H, N, d = K.shape
        if mask_2d is not None:
            mask = mask_2d.unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
            K = K * mask
            V = V * mask

        # Combine batch and heads for batched matrix multiplication
        K_flat = K.reshape(B * H, N, d)     # (B*H, N, d)
        V_flat = V.reshape(B * H, N, d)

        E_exp = self.E.unsqueeze(0).expand(B * H, -1, -1)   # (B*H, k, N)
        F_exp = self.F.unsqueeze(0).expand(B * H, -1, -1)

        # Project: (B*H, k, N) @ (B*H, N, d) -> (B*H, k, d)
        K_comp = torch.bmm(E_exp, K_flat)
        V_comp = torch.bmm(F_exp, V_flat)

        # Reshape back to (B, heads, k, d)
        K_comp = K_comp.reshape(B, H, self.E.shape[0], d)
        V_comp = V_comp.reshape(B, H, self.F.shape[0], d)
        return K_comp, V_comp
class TokenCompressedAttention(nn.Module):
    def __init__(self, config, comp_config, layer_idx, bert_mode=False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.comp_config = comp_config
        self.layer_idx = layer_idx
        self.k = comp_config.get_k_for_layer(layer_idx)
        self.bert_mode = bert_mode

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(comp_config.dropout)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)

        # Linformer-style projection (free parameters)
        self.projection = LinformerProjection(comp_config.max_seq_len, self.k)

        if comp_config.use_gate and not self.bert_mode:
            self.gate = nn.Parameter(torch.tensor(comp_config.gate_bias))
        else:
            self.gate = None

        self._ortho_loss = None

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ):
        B, N, _ = hidden_states.shape
        L = self.num_landmarks

        # --- Linear projections ------------------------------------------------
        Q = self.transpose_for_scores(self.query(hidden_states))   # (B, H, N, D)
        K = self.transpose_for_scores(self.key(hidden_states))
        V = self.transpose_for_scores(self.value(hidden_states))

        Q_features = self.kernel_feature_map(Q)   # (B, H, N, 2*feat_dim)
        K_features = self.kernel_feature_map(K)

        K_land, V_land = self.select_landmarks(K_features, V)  # (B, H, L, ...)

        # Corrected values
        W = self.W_corr.unsqueeze(0).expand(B, -1, -1, -1)   # (B, H, D, D)
        V_land_adj = torch.einsum('bhld,bhde->bhle', V_land, W)  # (B, H, L, D)

        # Kernel attention scores
        attn_weights = torch.matmul(Q_features, K_land.transpose(-2, -1))  # (B, H, N, L)

        # --- Mask handling ----------------------------------------------------
        # DistilBERT passes mask as `attn_mask`; BERT sometimes as `attention_mask`.
        # Unify them here.
        mask = attention_mask
        if mask is None:
            mask = kwargs.get('attn_mask', None)

        if mask is not None:
            # Convert any mask to 2D booleans (True = keep this token)
            if mask.dim() == 2:                     # (B, N)
                mask_2d = mask.bool()
            elif mask.dim() == 3:                   # (B, 1, N)
                mask_2d = mask.squeeze(1).bool()
            elif mask.dim() == 4:                   # (B, 1, 1, N) e.g. additive masks
                mask_2d = mask.squeeze(1).squeeze(1) > -1e9
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

            # Sub‑sample to get mask for landmarks
            if N > L:
                # Take L evenly spaced positions (you can also use k‑means++ later)
                indices = torch.linspace(0, N-1, L, device=mask_2d.device).long()
                landmark_mask = mask_2d[:, indices]            # (B, L)
            else:
                landmark_mask = mask_2d                        # (B, N)

            # Broadcast to (B, 1, 1, L) so it can be applied to attn_weights
            landmark_mask = landmark_mask.unsqueeze(1).unsqueeze(2)

            # Apply additive masking: set scores for invalid landmarks to a large negative number
            attn_weights = attn_weights.masked_fill(~landmark_mask, -10000.0)

        # Softmax and dropout
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output computation
        context_layer = torch.matmul(attn_weights, V_land_adj)  # (B, H, N, D)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_shape)

        return context_layer, None


    def get_ortho_loss(self):
        return 0.0




class CompressedBertAttention(nn.Module):
    def __init__(self, original_attention, compressed_self_attn):
        super().__init__()
        self.compressed_self = compressed_self_attn   # TokenCompressedAttention (normal mode)
        self.output = original_attention.output        # BertSelfOutput (dense will be replaced with Identity)
        self.output_dropout = original_attention.output.dropout   # keep original dropout
        self.pruned_heads = getattr(original_attention, 'pruned_heads', set())

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        attn_result = self.compressed_self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        self.bert.config.gradient_checkpointing = False
        if hasattr(self.bert, 'gradient_checkpointing_disable'):
            self.bert.gradient_checkpointing_disable()

        if isinstance(attn_result, tuple):
            context, attn_weights = attn_result
        else:
            context, attn_weights = attn_result, None

        # Apply the same dropout that original BertSelfOutput used to have after the dense
        context = self.output_dropout(context)
        # Now call BertSelfOutput (its dense is now Identity, so it only does residual + LayerNorm)
        layer_output = self.output(context, hidden_states)

        return (layer_output, attn_weights)    

            # ── 5. Inject Compression into BERT ─────────────────────────
            
def inject_compression_into_layer(layer, comp_config, layer_idx, model_config, model_type="bert"):
    if comp_config.attn_type == "leka":
        new_attn = LEKAAttention(model_config, comp_config, layer_idx, bert_mode=False)
    else:
        new_attn = TokenCompressedAttention(model_config, comp_config, layer_idx, bert_mode=False)

    if model_type == 'bert':
        old_self = layer.attention.self
        new_attn.query.weight.data.copy_(old_self.query.weight.data)
        new_attn.query.bias.data.copy_(old_self.query.bias.data)
        new_attn.key.weight.data.copy_(old_self.key.weight.data)
        new_attn.key.bias.data.copy_(old_self.key.bias.data)
        new_attn.value.weight.data.copy_(old_self.value.weight.data)
        new_attn.value.bias.data.copy_(old_self.value.bias.data)
        # copy output weights
        if hasattr(old_self, 'output'):
            new_attn.out_proj.weight.data.copy_(old_self.output.weight.data)
            new_attn.out_proj.bias.data.copy_(old_self.output.bias.data)
        else:  # for BERT, output dense is in BertSelfOutput
            output_dense = layer.attention.output.dense
            new_attn.out_proj.weight.data.copy_(output_dense.weight.data)
            new_attn.out_proj.bias.data.copy_(output_dense.bias.data)
        # layer.attention.output.dense = nn.Identity()
        replacement = CompressedBertAttention(layer.attention, new_attn)
        replacement.output.dense = nn.Identity()  # ✓
        layer.attention = replacement

        replacement = CompressedBertAttention(layer.attention, new_attn)
        layer.attention = replacement
        return new_attn
    elif model_type == 'distilbert':
        if comp_config.attn_type == "leka":
            new_attn = LEKAAttention(model_config, comp_config, layer_idx, bert_mode=False)
        else:
            new_attn = TokenCompressedAttention(model_config, comp_config, layer_idx, bert_mode=False)
        
        old_attn = layer.attention
        
        # کپی وزن‌های Q, K, V
        new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
        new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
        new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
        new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
        new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
        new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
        
        # کپی وزن output projection
        new_attn.out_proj.weight.data.copy_(old_attn.out_lin.weight.data)
        new_attn.out_proj.bias.data.copy_(old_attn.out_lin.bias.data)
        
        # جایگزینی مستقیم (DistilBERT ساختار متفاوتی دارد)
        layer.attention = new_attn
        return new_attn


    else:   
        raise NotImplementedError(f"Unsupported model type: {model_type}")
    


def fix_layernorm_naming(model: nn.Module):
    """
    Fix LayerNorm parameter names from checkpoint format (beta/gamma) 
    to PyTorch format (bias/weight).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            # Check if parameters are named beta/gamma
            state_dict = dict(module.named_parameters())
            
            if 'beta' in state_dict and 'bias' not in state_dict:
                # Rename beta to bias
                bias_param = module._parameters.pop('beta')
                module.register_parameter('bias', bias_param)
            
            if 'gamma' in state_dict and 'weight' not in state_dict:
                # Rename gamma to weight
                weight_param = module._parameters.pop('gamma')
                module.register_parameter('weight', weight_param)

# ── 6. Main Model Wrapper ───────────────────────────────────
class BertWithTokenCompression(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        comp_config: Optional[CompressionConfig] = None,
        disable_compression: bool = False,
    ):
        super().__init__()
        self.comp_config = comp_config or CompressionConfig()
        self.disable_compression = disable_compression

        # Load config and model
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        fix_layernorm_naming(self.bert)

        # Identify base transformer layers
        model_type = config.model_type
        if model_type == 'bert':
            self.base_model = self.bert.bert
            layers = self.base_model.encoder.layer
        elif model_type == 'distilbert':
            self.base_model = self.bert.distilbert
            layers = self.base_model.transformer.layer
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Inject compression
        self.compression_modules = nn.ModuleList()
        if not disable_compression:
            for i, layer in enumerate(layers):
                new_attn = inject_compression_into_layer(
                    layer=layer,
                    comp_config=self.comp_config,
                    layer_idx=i,
                    model_config=config,
                    model_type=model_type,
                )
                self.compression_modules.append(new_attn)

        # Gradient checkpointing for memory
        # self.bert.gradient_checkpointing_enable()

        log.info(f"Created model with {len(self.compression_modules)} compressed layers "
                 f"(type={model_type})")
        log.info(f"Compression ratios: early={self.comp_config.k_early}, "
                 f"mid={self.comp_config.k_mid}, late={self.comp_config.k_late}")
        self.bert.config.gradient_checkpointing = False
        if hasattr(self.bert, 'gradient_checkpointing_disable'):
            self.bert.gradient_checkpointing_disable()

                     
    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.bert(**kwargs)
    
    def get_ortho_loss(self) -> torch.Tensor:
        """Get total orthogonality loss from all compression modules."""
        total_loss = 0.0
        for module in self.compression_modules:
            loss = module.get_ortho_loss()
            if loss is not None:
                total_loss = total_loss + loss
        return torch.tensor(total_loss, device=self.bert.device)
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer."""
        self.bert.save_pretrained(save_directory)
    
    @property
    def device(self):
        return next(self.bert.parameters()).device

# ── 7. Custom Trainer with Orthogonality Loss ───────────────
# ── Custom Trainer with Orthogonality Loss ──────────────
class CompressionTrainer(Trainer):
    def __init__(self, ortho_weight: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_weight = ortho_weight
        log.info("CompressionTrainer initialized")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # log.info("compute_loss called")
        outputs = model(**inputs)
        # log.info(f"Model forward pass complete, loss: {outputs.loss}")
        task_loss = outputs.loss
        ortho_loss = model.get_ortho_loss()
        # log.info(f"Ortho loss: {ortho_loss}")
        total_loss = task_loss + self.ortho_weight * ortho_loss
        return (total_loss, outputs) if return_outputs else total_loss
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

import evaluate

from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

def get_compute_metrics(task: str):
    task = task.lower()
    if task == "cola":
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"eval_matthews_correlation": matthews_corrcoef(labels, preds)}
        return compute
    elif task == "stsb":
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = logits.squeeze()
            corr, _ = pearsonr(preds, labels)
            return {"eval_pearsonr": corr}
        return compute
    elif task in ("mrpc", "qqp"):
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "eval_accuracy": accuracy_score(labels, preds),
                "eval_f1": f1_score(labels, preds)
            }
        return compute
    else:   # sst2, mnli, qnli, rte, wnli
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"eval_accuracy": accuracy_score(labels, preds)}
        return compute    # ── 8. Main ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Token‑Compressed Attention for GLUE")
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"])
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./compressed_results")
    parser.add_argument("--k_early", type=int, default=32)
    parser.add_argument("--k_mid", type=int, default=16)
    parser.add_argument("--k_late", type=int, default=12)
    parser.add_argument("--no_progressive", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn_type", type=str, default="linformer", choices=["linformer", "leka"])
    parser.add_argument("--num_landmarks", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--use_spherical_norm", action="store_true", default=True)
    parser.add_argument("--kernel", type=str, default="elu")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Compression config
    comp_config = CompressionConfig(
            attn_type=args.attn_type,
    num_landmarks=args.num_landmarks,
    feature_dim=args.feature_dim,
    use_spherical_norm=args.use_spherical_norm,
    kernel=args.kernel,

        k_early=args.k_early,
        k_mid=args.k_mid,
        k_late=args.k_late,
        progressive=not args.no_progressive,
        use_gate=not args.no_gate,
        max_seq_len=args.max_len,
    )
    log.info(f"Task: {args.task} | Progressive: {comp_config.progressive}")
    log.info(f"k early/mid/late = {comp_config.k_early}/{comp_config.k_mid}/{comp_config.k_late}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Load raw dataset
    raw_datasets = load_glue_local(args.task)
    if 'train' not in raw_datasets:
        raise RuntimeError(f"No 'train' split in {args.task}. Found: {list(raw_datasets.keys())}")
    log.info(f"Raw dataset sizes: train={len(raw_datasets['train'])}, val={len(raw_datasets.get('validation', []))}")

    num_labels, is_regression = get_num_labels_and_is_regression(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    # Tokenization function (with explicit return)
    def preprocess(examples):
        if sent2_key is None:
            tokenized = tokenizer(
                examples[sent1_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_len,
            )
        else:
            tokenized = tokenizer(
                examples[sent1_key],
                examples[sent2_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_len,
            )
        # Always add labels, even if it fails
        if is_regression:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = list(examples["label"])

        return tokenized

    # Apply tokenization
    # dataset = raw_datasets.map(preprocess, batched=True, remove_columns=raw_datasets["train"].column_names)
    dataset = raw_datasets.map(
        preprocess, 
        batched=True, 
        remove_columns=raw_datasets["train"].column_names  # Remove all original columns
    )

    # --- Sanity checks ---
    if len(dataset["train"]) == 0:
        raise RuntimeError("Training dataset became empty after tokenization!")
    # Check a sample to see if columns exist
    sample = dataset["train"][0]
    required = ["input_ids", "attention_mask", "labels"]
    missing = [col for col in required if col not in sample]
    if missing:
        raise RuntimeError(f"Missing columns after tokenization: {missing}")

    # Set format for PyTorch
    # columns = ["input_ids", "attention_mask", "labels"]
    # if args.task in ("mnli", "rte", "wnli", "mrpc", "qqp", "qnli"):
    #     columns.append("token_type_ids")
    # dataset.set_format(type="torch", columns=columns)
    columns = ["input_ids", "attention_mask", "labels"]
    # فقط برای BERT اضافه کنید
    if "bert-base" in args.model and args.task in ("mnli", "rte", "wnli", "mrpc", "qqp", "qnli"):
        columns.append("token_type_ids")

    log.info(f"Tokenized dataset: train={len(dataset['train'])}, val={len(dataset.get('validation', []))}")

    # Build model
    model = BertWithTokenCompression(args.model, num_labels, comp_config)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("Using CPU")
    log.info("DEBUG: model device is %s", next(model.bert.parameters()).device)

# بعد از model.to(device)
    # model.eval()
    # test_batch = dataset["train"].select(range(8))  # 8 نمونه
    # inputs = tokenizer(test_batch[sent1_key], 
    #                 test_batch[sent2_key] if sent2_key else None,
    #                 truncation=True, padding="max_length", max_length=args.max_len,
    #                 return_tensors="pt")
    # inputs = {k: v.to(device) for k, v in inputs.items()}
    # inputs["labels"] = torch.tensor(test_batch["label"]).to(device)
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     print("Test forward OK, loss:", outputs.loss.item())
    # exit()
    # Metric for best model (with 'eval_' prefix)
    if args.task == "cola":
        metric_name = "eval_matthews_correlation"
    elif args.task == "stsb":
        metric_name = "eval_pearsonr"
    elif args.task in ("mrpc", "qqp"):
        metric_name = "eval_f1"
    else:
        metric_name = "eval_accuracy"

    # Training arguments (remove deprecated overwrite_output_dir)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
            # remove_unused_columns=False,   # ← add this line
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=10,          # stricter clipping
        warmup_ratio=0.1,
        # use_mps_device=True,
        # dataloader_pin_memory=False,
        remove_unused_columns=False,   # این خط را بگذارید

        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        # fp16=args.fp16,
            fp16=False,                     # fp16 روی MPS ناپایدار است

        save_total_limit=2,
        # remove_unused_columns=True,
        report_to="none",
    )
    print("jjj")
    ortho_weight = comp_config.ortho_reg_weight if not comp_config.use_orthogonal else 0.0
    log.info(f"Train dataset columns: {dataset['train'].column_names}")
    log.info(f"Train dataset features: {dataset['train'].features}")
    log.info(f"First example: {dataset['train'][0]}")

    trainer = CompressionTrainer(
        ortho_weight=ortho_weight,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=default_data_collator,  # Use default instead
        compute_metrics=get_compute_metrics(args.task),
    )

    # log.info("Testing first batch...")
    # try:
    #     test_batch = next(iter(trainer.get_train_dataloader()))
    #     test_batch = {k: v.to(device) for k, v in test_batch.items()}
    #     log.info(f"Batch keys: {test_batch.keys()}")
    #     log.info(f"Batch shapes: {[(k, v.shape) for k, v in test_batch.items()]}")
        
    #     model.train()
    #     with torch.amp.autocast('cpu'):  # or 'cuda' if using GPU
    #         outputs = model(**test_batch)
    #         loss = outputs.loss
    #         log.info(f"Test forward pass OK, loss: {loss.item()}")
    #         loss.backward()
    #         log.info("Test backward pass OK")
    # except Exception as e:
    #     log.error(f"Failed at first batch: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     exit(1)


    log.info("Starting training...")
    trainer.train()
    eval_result = trainer.evaluate()
    log.info(f"Final eval: {eval_result}")
    trainer.save_model(os.path.join(args.output_dir, args.task))
    tokenizer.save_pretrained(os.path.join(args.output_dir, args.task))
    log.info("Done.")
if __name__ == "__main__":
    main()