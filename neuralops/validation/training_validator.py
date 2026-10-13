"""End-to-end training validation on tiny data."""
import torch

class TrainingValidator:
    @staticmethod
    def overfit_test(model, batch, n_samples=8, steps=50, lr=1e-3, device='cpu'):
        print(f"\n=== Overfit test ({n_samples} samples, {steps} steps) ===")
        small_batch = {k: v[:n_samples].to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        losses = []
        model.train()
        for step in range(steps):
            out = model(**small_batch)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            if step % 10 == 0:
                print(f'  step {step}: loss = {loss.item():.4f}')
        if losses[-1] < losses[0] * 0.5:
            print(f'  ✅ Overfit OK: {losses[0]:.4f} → {losses[-1]:.4f}')
            return True
        else:
            print(f'  ❌ Overfit FAIL: {losses[0]:.4f} → {losses[-1]:.4f} (did not decrease by 50%)')
            return False
