"""Benchmark training throughput."""
import torch
import time

class SpeedValidator:
    @staticmethod
    def benchmark(model, batch, device, warmup=3, iters=10):
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for _ in range(warmup):
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        bs = batch['input_ids'].size(0)
        print(f'  Speed: {dt*1000:.2f} ms/step, Throughput: {bs/dt:.1f} samples/sec')
        return dt
