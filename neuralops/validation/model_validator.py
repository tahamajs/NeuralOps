"""Validate model construction and forward pass."""
import torch

class ModelValidator:
    def __init__(self, model, input_shape, device='cpu'):
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.errors = []

    def validate(self):
        print(f"\n=== Validating model ===")
        try:
            self.model = self.model.to(self.device)
            print(f"  ✅ Moved to {self.device}")
        except Exception as e:
            self.errors.append(f"Move to device failed: {e}")
            return False

        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  Total params: {n_params/1e6:.2f}M, Trainable: {n_trainable/1e6:.2f}M")

        try:
            self.model.eval()
            dummy = torch.randint(0, 1000, self.input_shape, device=self.device)
            with torch.no_grad():
                out = self.model(dummy)
            logits = out.logits if hasattr(out, 'logits') else out[0]
            print(f"  ✅ Forward pass: output shape {logits.shape}")
            if torch.isnan(logits).any():
                self.errors.append('NaN in forward output!')
            if torch.isinf(logits).any():
                self.errors.append('Inf in forward output!')
        except Exception as e:
            self.errors.append(f'Forward failed: {e}')
            return False

        try:
            self.model.train()
            # Try classification label shape (B,) first, fallback to token-level (B, L)
            try:
                labels = torch.zeros(self.input_shape[0], dtype=torch.long, device=self.device)
                out = self.model(dummy, labels=labels)
            except Exception:
                labels = torch.zeros(self.input_shape[0], self.input_shape[1], dtype=torch.long, device=self.device)
                out = self.model(dummy, labels=labels)
            out.loss.backward()
            print(f"  ✅ Backward pass: loss {out.loss.item():.4f}")
            self.model.zero_grad()
        except Exception as e:
            self.errors.append(f'Backward failed: {e}')

        n_nan_grads = sum(1 for p in self.model.parameters() if p.grad is not None and torch.isnan(p.grad).any())
        if n_nan_grads > 0:
            self.errors.append(f'{n_nan_grads} params with NaN grads')

        print(f"  Errors: {len(self.errors)}")
        return len(self.errors) == 0
