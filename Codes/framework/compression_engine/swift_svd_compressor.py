"""Swift‑SVD Activation‑aware via cross‑covariance. Registered as 'swift_svd_act'."""

import torch
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class SwiftSVDActivationCompressor(BaseCompressor):
    def __init__(self, model, tokenizer, rank=64, calibration_texts=None, **kwargs):
        super().__init__(model, name="swift_svd_act", **kwargs)
        self.tokenizer = tokenizer
        self.rank = rank
        self.calibration_texts = calibration_texts or ["Sample text."]

    def compress(self) -> Any:
        """Compress the target module using this strategy."""
        self.model.eval()
        activations = {}
        hooks = []

        def make_hook(name) -> Any:
            """Make hook."""
            def hook(module, input, output) -> None:
                """Hook."""
                activations[name] = (input[0].detach(), output.detach())

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                hooks.append(module.register_forward_hook(make_hook(name)))

        inputs = self.tokenizer(
            self.calibration_texts, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and name in activations:
                inp, out = activations[name]
                # Flatten activations
                inp_flat = inp.view(-1, inp.size(-1)).float()
                out_flat = out.view(-1, out.size(-1)).float()

                # Cross‑covariance: shape (out_features, in_features)
                C = out_flat.t() @ inp_flat / inp_flat.size(0)

                # Truncated SVD of the cross‑covariance
                U, S, Vt = torch.linalg.svd(C, full_matrices=False)
                r = min(self.rank, min(C.shape))

                # Low‑rank weight: (out_features, in_features)
                W_new = (U[:, :r] * S[:r].unsqueeze(0)) @ Vt[:r, :]

                module.weight.data = W_new.to(dtype=module.weight.dtype)

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("swift_svd_act")
def swift_svd_act_factory(model, comp_config) -> Any:
    """Swift svd act factory."""
    rank = getattr(comp_config, "swift_svd_act_rank", 64)
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("swift_svd_act requires tokenizer attached as model.tokenizer")
    compressor = SwiftSVDActivationCompressor(model, tokenizer, rank)
    compressed_model, _ = compressor.compress()
    return compressed_model
