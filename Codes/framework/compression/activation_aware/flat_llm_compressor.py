import torch
import numpy as np

class FLATLLMCompressor:
    """
    Applies PCA to activation outputs of each layer to identify important dimensions.
    Prunes output dimensions of the preceding linear layer accordingly.
    """
    def __init__(self, model, tokenizer, keep_ratio=0.7, calibration_texts=None):
        self.model = model
        self.tokenizer = tokenizer
        self.keep_ratio = keep_ratio
        self.calibration_texts = calibration_texts or ["The quick brown fox jumps over the lazy dog."]

    def compress(self):
        self.model.eval()
        # Register hooks to capture activations
        activations = {}
        hooks = []
        def hook_fn(module, input, output, name):
            activations[name] = output.detach().cpu()

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))

        # Forward pass with calibration data
        inputs = self.tokenizer(self.calibration_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        # For each linear layer, prune output dimensions
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in activations:
                act = activations[name]  # (batch*seq, out_features)
                # Flatten to 2D
                act_flat = act.view(-1, act.size(-1)).numpy()
                # PCA on activations
                cov = np.cov(act_flat.T)
                eigenvalues, _ = np.linalg.eigh(cov)
                # Keep top dimensions
                total_dims = len(eigenvalues)
                keep_dims = int(total_dims * self.keep_ratio)
                # Sort eigenvalues descending
                idx = np.argsort(eigenvalues)[::-1]
                keep_idx = idx[:keep_dims]
                # Prune weight and bias
                W = module.weight.data  # (out_features, in_features)
                module.weight.data = W[keep_idx, :]
                if module.bias is not None:
                    module.bias.data = module.bias.data[keep_idx]
        return self.model