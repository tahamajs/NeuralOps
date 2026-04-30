# framework/compression/advanced_methods.py

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path

# --------------- Existing attention replacement (SVD‑based) ---------------
from ..attentions.compressed_attention import CompressedAttentionLayer

# --------------- New attention replacements ---------------
from ..attentions.nystrom_attention import NystromAttention
from ..attentions.tpa_attention import TPAttention
from ..attentions.primal_attention import PrimalAttention
from ..attentions.hamming_attention import HammingAttention
from ..attentions.rala_attention import RALALinearAttention
from ..attentions.nha_attention import HybridAttention

# --------------- Post‑training compressors ---------------
from .slicegpt import SliceGPTCompressor
from .sparsegpt import SparseGPTPruner
from .flap import FLAPCompressor
from .oats import OATSCompressor
from .mamba_shedder import MambaShedderCompressor
from .come import CoMeCompressor
from .rocketkv import RocketKVCompressor
# Dobi‑SVD layer is a replacement class, not a compressor, so imported separately
from .dobi_svd import DobiSVDLinearLayer


class AdvancedZeroShotCompressor:
    def __init__(self, method: str, base_threshold: float = 0.90, min_rank: int = 4):
        self.method = method
        self.base_threshold = base_threshold
        self.min_rank = min_rank

    def _get_rank_from_energy(self, S, threshold):
        if len(S) == 0:
            return self.min_rank
        energy = np.cumsum(S) / np.sum(S)
        try:
            rank = np.argmax(energy >= threshold) + 1
        except:
            rank = self.min_rank
        return max(self.min_rank, min(rank, len(S)))

    def compress_model(self, model: nn.Module, method: str) -> Tuple[nn.Module, List[Dict], float]:
        print(f"🔧 Starting Advanced Method: {method}")
        compression_stats = []
        orig_total_params = sum(p.numel() for p in model.parameters())

        # -----------------------------------------------------------------
        # 1. METHODS THAT REPLACE MULTIHEAD ATTENTION WITH CUSTOM LAYERS
        # -----------------------------------------------------------------
        if method in [
            "standard_svd", "head_pruning", "layer_skipping", "quantized_svd",
            "blockwise_lowrank", "randomized_svd", "tensor_train_decomp",
            "cp_decomposition", "dynamic_token_pruning", "spectral_clustering_prune",
            "hash_quantization", "moe_sparsification", "hybrid_advanced",
            "linformer", "performer", "asvd", "svd_llm_v2", "nystrom", "tpa", "primal", "hamming",
            "rala", "nha"   # <-- added RALA and NHA
        ]:
            layers_to_compress = []
            for name, module in model.named_modules():
                if isinstance(module, nn.MultiheadAttention):
                    w_qkv = module.in_proj_weight.data.cpu().numpy()
                    w_o = module.out_proj.weight.data.cpu().numpy()
                    _, S_qkv, _ = np.linalg.svd(w_qkv, full_matrices=False)
                    _, S_out, _ = np.linalg.svd(w_o, full_matrices=False)
                    layers_to_compress.append({
                        'name': name,
                        'module': module,
                        'config': getattr(module, 'config', None),  # store config for RALA/NHA
                        'S_qkv': S_qkv,
                        'S_out': S_out,
                        'original_dim': module.embed_dim,
                        'index': len(layers_to_compress)
                    })

            total_layers = len(layers_to_compress)
            if total_layers == 0:
                return model, [], 0.0

            ranks_list = []
            if method == "standard_svd":
                ranks_list = [(self._get_rank_from_energy(info['S_qkv'], self.base_threshold),
                               self._get_rank_from_energy(info['S_out'], self.base_threshold))
                              for info in layers_to_compress]
            elif method == "head_pruning":
                half_dim = self.min_rank * 2
                ranks_list = [(half_dim, half_dim) for _ in layers_to_compress]
            elif method == "layer_skipping":
                skip_ratio = 0.3
                skip_count = int(total_layers * skip_ratio)
                for i, info in enumerate(layers_to_compress):
                    if i < total_layers - skip_count:
                        r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                    else:
                        r = self.min_rank
                    ranks_list.append((r, r))
            elif method == "quantized_svd":
                ranks_list = [(self._get_rank_from_energy(info['S_qkv'], 0.75),
                               self._get_rank_from_energy(info['S_out'], 0.75))
                              for info in layers_to_compress]
            elif method == "blockwise_lowrank":
                block_size = 4
                for i, info in enumerate(layers_to_compress):
                    block_idx = i // block_size
                    r = max(self.min_rank, int(self._get_rank_from_energy(info['S_qkv'], self.base_threshold) * (1 - 0.1 * block_idx)))
                    ranks_list.append((r, r))
            elif method == "randomized_svd":
                fixed_rank = 16
                ranks_list = [(fixed_rank, fixed_rank) for _ in layers_to_compress]
            elif method == "tensor_train_decomp":
                start_r = self._get_rank_from_energy(layers_to_compress[0]['S_qkv'], self.base_threshold)
                end_r = max(self.min_rank, int(start_r * 0.2))
                for i, info in enumerate(layers_to_compress):
                    progress = i / (total_layers - 1) if total_layers > 1 else 0
                    r = int(start_r * (end_r / start_r) ** (progress * 2))
                    r = max(self.min_rank, min(r, info['original_dim']))
                    ranks_list.append((r, r))
            elif method == "cp_decomposition":
                fixed_cp_rank = 6
                ranks_list = [(fixed_cp_rank, fixed_cp_rank) for _ in layers_to_compress]
            elif method == "dynamic_token_pruning":
                for i, info in enumerate(layers_to_compress):
                    r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                    if i > total_layers * 0.6:
                        r = max(self.min_rank, int(r * 0.6))
                    ranks_list.append((r, r))
            elif method == "spectral_clustering_prune":
                median_energy = np.median([np.mean(l['S_qkv']) for l in layers_to_compress])
                for info in layers_to_compress:
                    if np.mean(info['S_qkv']) < median_energy:
                        r = self.min_rank
                    else:
                        r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                    ranks_list.append((r, r))
            elif method == "hash_quantization":
                fixed_hash_rank = 8
                ranks_list = [(fixed_hash_rank, fixed_hash_rank) for _ in layers_to_compress]
            elif method == "moe_sparsification":
                for i, info in enumerate(layers_to_compress):
                    if i < total_layers // 2:
                        r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                    else:
                        r = max(self.min_rank, int(self._get_rank_from_energy(info['S_qkv'], self.base_threshold) * 0.3))
                    ranks_list.append((r, r))
            elif method == "hybrid_advanced":
                mid = total_layers // 2
                deep_start = int(total_layers * 0.7)
                for i, info in enumerate(layers_to_compress):
                    if i < mid:
                        start_r = self._get_rank_from_energy(layers_to_compress[0]['S_qkv'], self.base_threshold)
                        end_r = max(self.min_rank, int(start_r * 0.2))
                        progress = i / (mid - 1) if mid > 1 else 0
                        r = int(start_r * (end_r / start_r) ** (progress * 2))
                    elif i < deep_start:
                        r = self._get_rank_from_energy(info['S_qkv'], self.base_threshold)
                    else:
                        r = max(self.min_rank, int(r * 0.3))
                    ranks_list.append((r, r))
            elif method in [
                "linformer", "performer", "nystrom", "tpa", "primal", "hamming",
                "rala", "nha"
            ]:
                # Dummy ranks (not used for attention replacement)
                ranks_list = [(self.min_rank, self.min_rank) for _ in layers_to_compress]
            else:
                raise ValueError(f"Unknown method: {method}")

            compressed_count = 0
            for i, (info, (r_qkv, r_out)) in enumerate(zip(layers_to_compress, ranks_list)):
                try:
                    if method in [
                        "standard_svd", "head_pruning", "layer_skipping", "quantized_svd",
                        "blockwise_lowrank", "randomized_svd", "tensor_train_decomp",
                        "cp_decomposition", "dynamic_token_pruning", "spectral_clustering_prune",
                        "hash_quantization", "moe_sparsification", "hybrid_advanced"
                    ]:
                        new_layer = CompressedAttentionLayer(info['module'], r_qkv, r_out, method)
                    elif method == "linformer":
                        new_layer = LinformerAttention(info['module'], rank=self.min_rank)
                    elif method == "performer":
                        new_layer = PerformerAttention(info['module'])
                    elif method == "nystrom":
                        new_layer = NystromAttention(info['module'], rank=self.min_rank * 4)
                    elif method == "tpa":
                        new_layer = TPAttention(info['module'], rank_Q=1, rank_K=1, rank_V=1)
                    elif method == "primal":
                        new_layer = PrimalAttention(info['module'], rank=self.min_rank * 4)
                    elif method == "hamming":
                        new_layer = HammingAttention(info['module'], rank=self.min_rank)
                    # --------------- NEW ATTENTION METHODS ---------------
                    elif method == "rala":
                        # Use the module's config if available, else derive from attributes
                        config = info.get('config')
                        if config is None:
                            from types import SimpleNamespace
                            mod = info['module']
                            config = SimpleNamespace(
                                hidden_size=mod.embed_dim * mod.num_heads,
                                num_attention_heads=mod.num_heads,
                                hidden_act="relu",
                            )
                        new_layer = RALALinearAttention(config)
                        # Copy projection weights from the original module
                        new_layer.q_proj.load_state_dict(info['module'].in_proj_weight.chunk(3, dim=0)[0].reshape(-1, config.hidden_size).state_dict())
                        new_layer.k_proj.load_state_dict(info['module'].in_proj_weight.chunk(3, dim=0)[1].reshape(-1, config.hidden_size).state_dict())
                        new_layer.v_proj.load_state_dict(info['module'].in_proj_weight.chunk(3, dim=0)[2].reshape(-1, config.hidden_size).state_dict())
                        new_layer.out_proj.load_state_dict(info['module'].out_proj.weight.state_dict())
                    elif method == "nha":
                        config = info.get('config')
                        if config is None:
                            from types import SimpleNamespace
                            mod = info['module']
                            config = SimpleNamespace(
                                hidden_size=mod.embed_dim * mod.num_heads,
                                num_attention_heads=mod.num_heads,
                                hidden_act="relu",
                                nha_full_ratio=0.1,
                            )
                        new_layer = HybridAttention(config)
                        new_layer.q_proj.load_state_dict(info['module'].in_proj_weight.chunk(3, dim=0)[0].reshape(-1, config.hidden_size).state_dict())
                        new_layer.k_proj.load_state_dict(info['module'].in_proj_weight.chunk(3, dim=0)[1].reshape(-1, config.hidden_size).state_dict())
                        new_layer.v_proj.load_state_dict(info['module'].in_proj_weight.chunk(3, dim=0)[2].reshape(-1, config.hidden_size).state_dict())
                        new_layer.out_proj.load_state_dict(info['module'].out_proj.weight.state_dict())
                    else:
                        raise ValueError(f"Unsupported attention replacement method: {method}")

                    parent_name = '.'.join(info['name'].split('.')[:-1])
                    child_name = info['name'].split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, new_layer)
                    compressed_count += 1

                    orig_p = sum(p.numel() for p in info['module'].parameters())
                    new_p = sum(p.numel() for p in new_layer.parameters())
                    compression_stats.append({
                        'layer': info['name'],
                        'original_params': orig_p,
                        'compressed_params': new_p,
                        'ratio': new_p / orig_p,
                        'saved': orig_p - new_p,
                        'target_rank': r_qkv
                    })
                except Exception as e:
                    print(f"   ❌ Failed to compress {info['name']}: {e}")

            new_total_params = sum(p.numel() for p in model.parameters())
            saved = orig_total_params - new_total_params
            reduction_pct = (saved / orig_total_params) * 100
            print(f"✅ Attention replacement complete! Method: {method}, Layers: {compressed_count}/{total_layers}, Reduction: {reduction_pct:.2f}%")
            return model, compression_stats, reduction_pct

        # -----------------------------------------------------------------
        # 2. METHODS THAT USE POST‑TRAINING COMPRESSION (FULL MODEL)
        # -----------------------------------------------------------------
        elif method == "slicegpt":
            from .slicegpt import SliceGPTCompressor
            print(f"   ✂️ Applying SliceGPT (sparsity={self.base_threshold})...")
            compressor = SliceGPTCompressor(
                model=model,
                tokenizer=tokenizer,
                sparsity=self.base_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            compressed_model, _ = compressor.compress()
            model = compressed_model
            compression_stats = [{
                'layer': 'model',
                'original_params': orig_total_params,
                'compressed_params': sum(p.numel() for p in model.parameters()),
                'ratio': sum(p.numel() for p in model.parameters()) / orig_total_params,
                'saved': orig_total_params - sum(p.numel() for p in model.parameters()),
                'target_rank': 0
            }]
            reduction_pct = (1 - sum(p.numel() for p in model.parameters()) / orig_total_params) * 100
            return model, compression_stats, reduction_pct

        elif method == "sparsegpt":
            from .sparsegpt import SparseGPTPruner
            print(f"   ✂️ Applying SparseGPT (sparsity={self.base_threshold})...")
            pruner = SparseGPTPruner(
                model=model,
                tokenizer=tokenizer,
                sparsity=self.base_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            compressed_model, _ = pruner.prune()
            model = compressed_model
            compression_stats = [{
                'layer': 'model',
                'original_params': orig_total_params,
                'compressed_params': sum(p.numel() for p in model.parameters()),
                'ratio': sum(p.numel() for p in model.parameters()) / orig_total_params,
                'saved': orig_total_params - sum(p.numel() for p in model.parameters()),
                'target_rank': 0
            }]
            reduction_pct = (1 - sum(p.numel() for p in model.parameters()) / orig_total_params) * 100
            return model, compression_stats, reduction_pct

        elif method == "flap":
            from .flap import FLAPCompressor
            print(f"   ✂️ Applying FLAP (sparsity={self.base_threshold})...")
            compressor = FLAPCompressor(
                model=model,
                tokenizer=tokenizer,
                sparsity=self.base_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            compressed_model, _ = compressor.compress()
            model = compressed_model
            compression_stats = [{
                'layer': 'model',
                'original_params': orig_total_params,
                'compressed_params': sum(p.numel() for p in model.parameters()),
                'ratio': sum(p.numel() for p in model.parameters()) / orig_total_params,
                'saved': orig_total_params - sum(p.numel() for p in model.parameters()),
                'target_rank': 0
            }]
            reduction_pct = (1 - sum(p.numel() for p in model.parameters()) / orig_total_params) * 100
            return model, compression_stats, reduction_pct

        elif method == "oats":
            from .oats import OATSCompressor
            print(f"   ✂️ Applying OATS (sparsity={self.base_threshold})...")
            compressor = OATSCompressor(
                model=model,
                tokenizer=tokenizer,
                sparsity=self.base_threshold,
                rank_ratio=0.2,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            compressed_model, _ = compressor.compress()
            model = compressed_model
            compression_stats = [{
                'layer': 'model',
                'original_params': orig_total_params,
                'compressed_params': sum(p.numel() for p in model.parameters()),
                'ratio': sum(p.numel() for p in model.parameters()) / orig_total_params,
                'saved': orig_total_params - sum(p.numel() for p in model.parameters()),
                'target_rank': 0
            }]
            reduction_pct = (1 - sum(p.numel() for p in model.parameters()) / orig_total_params) * 100
            return model, compression_stats, reduction_pct

        elif method == "mamba_shedder":
            from .mamba_shedder import MambaShedderCompressor
            print(f"   ✂️ Applying Mamba‑Shedder (sparsity={self.base_threshold})...")
            compressor = MambaShedderCompressor(
                model=model,
                tokenizer=tokenizer,
                sparsity=self.base_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            # Needs a calibration loader – for simplicity, use a minimal dummy loader
            class DummyLoader:
                def __iter__(self):
                    yield {"input_ids": torch.randint(0, 100, (1, 128)), "attention_mask": torch.ones(1, 128)}
            calib_loader = DummyLoader()
            compressed_model = compressor.compress(calib_loader)
            model = compressed_model
            compression_stats = [{
                'layer': 'model',
                'original_params': orig_total_params,
                'compressed_params': sum(p.numel() for p in model.parameters()),
                'ratio': sum(p.numel() for p in model.parameters()) / orig_total_params,
                'saved': orig_total_params - sum(p.numel() for p in model.parameters()),
                'target_rank': 0
            }]
            reduction_pct = (1 - sum(p.numel() for p in model.parameters()) / orig_total_params) * 100
            return model, compression_stats, reduction_pct

        elif method == "dobi_svd":
            from .dobi_svd import DobiSVDLinearLayer
            print(f"   🧠 Replacing Linear layers with Dobi‑SVD (target ratio={self.base_threshold})...")
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear) and 'classifier' not in name:
                    new_layer = DobiSVDLinearLayer(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        bias=module.bias is not None,
                        init_rank_ratio=self.base_threshold
                    )
                    parent_name = '.'.join(name.split('.')[:-1])
                    child_name = name.split('.')[-1]
                    parent = model.get_submodule(parent_name) if parent_name else model
                    setattr(parent, child_name, new_layer)
            return model, [], 0.0

        elif method == "come":
            from .come import CoMeCompressor
            print(f"   🧩 Applying CoMe layer concatenation (sparsity={self.base_threshold})...")
            compressor = CoMeCompressor(
                model=model,
                tokenizer=tokenizer,
                sparsity=self.base_threshold,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            compressed_model, _ = compressor.compress()
            model = compressed_model
            compression_stats = [{
                'layer': 'model',
                'original_params': orig_total_params,
                'compressed_params': sum(p.numel() for p in model.parameters()),
                'ratio': sum(p.numel() for p in model.parameters()) / orig_total_params,
                'saved': orig_total_params - sum(p.numel() for p in model.parameters()),
                'target_rank': 0
            }]
            reduction_pct = (1 - sum(p.numel() for p in model.parameters()) / orig_total_params) * 100
            return model, compression_stats, reduction_pct

        elif method == "rocketkv":
            from .rocketkv import RocketKVCompressor
            print(f"   🚀 Applying RocketKV KV cache compression (budget={self.base_threshold})...")
            model_name = getattr(model.config, "_name_or_path", "llama3.1-8b-instruct")
            compressor = RocketKVCompressor(
                model_name=model_name,
                token_budget=int(self.base_threshold * 1024),
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            output_dir = compressor.compress()
            # RocketKV does not return a modified model object – it's a runtime method.
            compression_stats = [{
                'layer': 'KV cache',
                'original_params': 0,
                'compressed_params': 0,
                'ratio': 0.0,
                'saved': 0,
                'target_rank': self.base_threshold
            }]
            return model, compression_stats, 0.0

        elif method == "resformer":
            from ..soa.resformer_block import ResFormerBlock
            print(f"   🔄 Replacing SOABlock with ResFormerBlock...")
            for i, layer in enumerate(model.bert.encoder.layer):
                new_block = ResFormerBlock(model.config, model.basis)
                new_block.load_state_dict(layer.state_dict(), strict=False)
                model.bert.encoder.layer[i] = new_block
            return model, [], 0.0

        else:
            raise ValueError(f"Unknown method: {method}")
