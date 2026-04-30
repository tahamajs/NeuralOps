# framework/analysis/layerwise_analyzer.py
"""
Layer-wise analysis of transformer models.
"""

import torch
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import entropy
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class LayerwiseAnalyzer:
    """
    Analyze layer-wise properties of transformer models.
    
    Computes:
    - Cosine similarities between consecutive layers
    - Effective ranks (entropy-based) of layer representations
    - Intrinsic dimensions using MLE
    - Attention entropy (if attention maps available)
    
    Example:
        >>> analyzer = LayerwiseAnalyzer(model, tokenizer)
        >>> results = analyzer.analyze("This is a sample text.")
        >>> print(results['cosine_similarities'])
    """
    
    def __init__(self, model, tokenizer, device: Optional[str] = None):
        """
        Args:
            model: HuggingFace transformer model
            tokenizer: Corresponding tokenizer
            device: Device to run on ('cuda', 'mps', 'cpu')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or self._get_device()
        self.model.to(self.device)
        self.hidden_states = []
        self.attention_maps = []
        self.hooks = []
        
    def _get_device(self) -> str:
        """Automatically select best available device."""
        if torch.cuda.is_available():
            return 'cuda'
        if torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def register_hooks(self, capture_attention: bool = False):
        """Register forward hooks to capture hidden states."""
        self.hidden_states = []
        self.attention_maps = []
        
        def hook_fn(module, input, output, name):
            if isinstance(output, tuple):
                hidden = output[0].detach().cpu()
                self.hidden_states.append(hidden)
                if capture_attention and len(output) > 1 and output[1] is not None:
                    self.attention_maps.append(output[1].detach().cpu())
            else:
                self.hidden_states.append(output.detach().cpu())
        
        # Register hooks for each encoder layer
        for name, module in self.model.named_modules():
            if 'encoder.layer' in name or 'transformer.layer' in name:
                hook = module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )
                self.hooks.append(hook)
    
    def compute_cosine_similarities(self) -> List[float]:
        """
        Compute cosine similarity between consecutive layer representations.
        Higher similarity = less information change between layers.
        """
        if len(self.hidden_states) < 2:
            return []
        
        similarities = []
        for i in range(len(self.hidden_states) - 1):
            h1 = self.hidden_states[i]
            h2 = self.hidden_states[i + 1]
            
            # Average over batch and sequence dimensions
            h1_flat = h1.mean(dim=(0, 1)).numpy()
            h2_flat = h2.mean(dim=(0, 1)).numpy()
            
            # Handle potential NaN or inf
            if np.any(np.isnan(h1_flat)) or np.any(np.isnan(h2_flat)):
                sim = 0.0
            else:
                try:
                    sim = 1 - cosine(h1_flat, h2_flat)
                except Exception:
                    sim = 0.0
            
            similarities.append(max(0.0, min(1.0, sim)))
        
        return similarities
    
    def compute_effective_ranks(self) -> List[float]:
        """
        Compute effective rank (entropy-based) of each layer's representation.
        Lower effective rank = more compressible.
        """
        effective_ranks = []
        
        for hs in self.hidden_states:
            # Flatten to 2D: (batch * seq_len, hidden_dim)
            H = hs.reshape(-1, hs.shape[-1]).numpy()
            
            # Compute singular values
            try:
                s = np.linalg.svd(H, compute_uv=False)
                s = s[s > 1e-8]  # Remove near-zero singular values
                
                if len(s) == 0:
                    effective_ranks.append(1.0)
                    continue
                
                # Normalize to probability distribution
                s_norm = s / s.sum()
                
                # Compute entropy
                ent = entropy(s_norm + 1e-10)
                
                # Effective rank = exp(entropy)
                effective_rank = np.exp(ent)
                effective_ranks.append(effective_rank)
                
            except Exception:
                effective_ranks.append(float(hs.shape[-1]))
        
        return effective_ranks
    
    def compute_intrinsic_dimension(self, k: int = 5) -> List[float]:
        """
        Estimate intrinsic dimension of each layer's representation using MLE.
        Lower intrinsic dimension = more compressible.
        """
        def estimate_id(data: np.ndarray) -> float:
            N, D = data.shape
            if N < k + 1:
                return float(D)
            
            # Subsample if too large
            if N > 2000:
                indices = np.random.choice(N, 2000, replace=False)
                data = data[indices]
                N = 2000
            
            # Compute pairwise distances
            from scipy.spatial.distance import pdist, squareform
            distances = squareform(pdist(data))
            
            # For each point, get distance to k-th neighbor
            kth_distances = []
            for i in range(N):
                sorted_dists = np.sort(distances[i, :])
                # Skip self-distance (which is 0)
                kth_dists = sorted_dists[1:k+1]
                kth_distances.append(kth_dists[-1])
            
            kth_distances = np.array(kth_distances)
            valid = kth_distances > 0
            if not valid.any():
                return float(D)
            
            # MLE estimate (simplified)
            log_ratios = np.log(kth_distances[valid] / (kth_distances[valid] + 1e-8))
            id_estimate = -1.0 / np.mean(log_ratios)
            
            return max(1.0, min(float(D), id_estimate))
        
        ids = []
        for hs in self.hidden_states:
            H = hs.reshape(-1, hs.shape[-1]).numpy()
            id_val = estimate_id(H)
            ids.append(id_val)
        
        return ids
    
    def analyze(self, text: str, capture_attention: bool = False) -> Dict[str, Any]:
        """
        Run complete analysis on a given text.
        
        Args:
            text: Input text to analyze
            capture_attention: Whether to capture attention weights (slower)
            
        Returns:
            Dictionary with analysis results
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Clear previous states and register hooks
        self.hidden_states = []
        self.attention_maps = []
        self._remove_hooks()
        self.register_hooks(capture_attention=capture_attention)
        
        # Forward pass
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=capture_attention)
        
        # Compute metrics
        results = {
            'num_layers': len(self.hidden_states),
            'hidden_dim': self.hidden_states[0].shape[-1] if self.hidden_states else 0,
            'cosine_similarities': self.compute_cosine_similarities(),
            'effective_ranks': self.compute_effective_ranks(),
            'intrinsic_dimensions': self.compute_intrinsic_dimension(),
        }
        
        if capture_attention and self.attention_maps:
            results['attention_entropy'] = self._compute_attention_entropy()
        
        # Clean up
        self._remove_hooks()
        
        return results
    
    def _compute_attention_entropy(self) -> List[float]:
        """Compute average attention entropy per layer."""
        entropies = []
        
        for attn in self.attention_maps:
            if attn.ndim < 4:
                continue
            
            attn_without_padding = attn + 1e-10
            entropy_per_head = -torch.sum(attn_without_padding * torch.log(attn_without_padding), dim=-1)
            avg_entropy = entropy_per_head.mean().item()
            entropies.append(avg_entropy)
        
        return entropies
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks = []
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save analysis results to JSON file."""
        results_json = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray):
                results_json[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                results_json[k] = v.cpu().tolist()
            else:
                results_json[k] = v
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        print(f"Saved results to {output_path}")