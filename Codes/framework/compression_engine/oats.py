"""OATS: Outlier-Aware Training Sparsification. Registered as 'oats'."""

import torch
import torch.nn as nn
import numpy as np
from framework.core.base import BaseCompressor
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

class OATSCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, sparsity=0.5, rank_ratio=0.2, num_iters=5, device="cuda", **kwargs):
        super().__init__(model, name="oats", **kwargs)
        self.tokenizer = tokenizer
        self.sparsity = sparsity
        self.rank_ratio = rank_ratio
        self.num_iters = num_iters
        self.device = device

    def _compute_second_moment(self, inputs):
        return (inputs.T @ inputs) / inputs.shape[0]

    def _oats_decompose(self, W, M, sparsity, rank_ratio, num_iters):
        L_chol = torch.linalg.cholesky(M)
        W_scaled = W @ L_chol.T
        U, s, Vt = np.linalg.svd(W_scaled.cpu().numpy(), full_matrices=False)
        rank = max(1, int(rank_ratio * min(W.shape)))
        L = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
        R = W_scaled.cpu().numpy() - L
        flat_R = R.flatten()
        n_keep = int((1 - sparsity) * len(flat_R))
        threshold = np.sort(np.abs(flat_R))[-n_keep]
        S = R * (np.abs(R) >= threshold)

        for _ in range(num_iters):
            U, s, Vt = np.linalg.svd(R - S, full_matrices=False)
            rank = max(1, int(rank_ratio * min(W.shape)))
            L = U[:, :rank] @ np.diag(s[:rank]) @ Vt[:rank, :]
            R_new = R - L
            flat_R_new = R_new.flatten()
            n_keep = int((1 - sparsity) * len(flat_R_new))
            threshold = np.sort(np.abs(flat_R_new))[-n_keep]
            S = R_new * (np.abs(R_new) >= threshold)

        W_compressed = torch.from_numpy(L + S).float() @ L_chol.T
        return W_compressed.to(W.device)

    def compress(self):
        print(f"✂️ Applying OATS (sparsity={self.sparsity}, rank_ratio={self.rank_ratio})...")
        self.model.eval()
        calib_texts = ["The quick brown fox jumps over the lazy dog."] * 100
        inputs = self.tokenizer(calib_texts, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs.input_ids.to(self.device)
        with torch.no_grad():
            embeds = self.model.get_input_embeddings()(input_ids)
            flat_inputs = embeds.view(-1, embeds.shape[-1])
        M = self._compute_second_moment(flat_inputs)

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and 'classifier' not in name:
                W = module.weight.data
                W_comp = self._oats_decompose(W, M, self.sparsity, self.rank_ratio, self.num_iters)
                module.weight.data = W_comp
        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()
    


@COMPRESSOR_REGISTRY.register("oats")
def oats_compressor_factory(model, comp_config):
    sparsity = getattr(comp_config, 'oats_sparsity', 0.5)
    rank_ratio = getattr(comp_config, 'oats_rank_ratio', 0.2)
    num_iters = getattr(comp_config, 'oats_num_iters', 5)
    # Note: tokenizer must be attached to model or passed separately
    # For the test, assume model.tokenizer is set
    tokenizer = getattr(model, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError("OATS compressor requires tokenizer attached as model.tokenizer")
    compressor = OATSCompressor(model, tokenizer, sparsity, rank_ratio, num_iters)
    compressed_model, _ = compressor.compress()
    return compressed_model
