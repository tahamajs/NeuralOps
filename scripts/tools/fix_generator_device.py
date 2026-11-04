with open("examples/_shared/offline_data.py", "r") as f:
    code = f.read()

# PyTorch requires generator device to match tensor device when generator is passed
old_code = """def toy_dataloader(
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 1000,
    num_batches: int = 4,
    device: str = "cpu",
    seed: int = 42,
) -> Iterator[dict[str, torch.Tensor]]:
    \"\"\"Generates reproducible dummy batches with deterministic integer seeds.\"\"\"
    generator = torch.Generator().manual_seed(seed)
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator, device=device)"""

new_code = """def toy_dataloader(
    batch_size: int = 4,
    seq_len: int = 32,
    vocab_size: int = 1000,
    num_batches: int = 4,
    device: str = "cpu",
    seed: int = 42,
) -> Iterator[dict[str, torch.Tensor]]:
    \"\"\"Generates reproducible dummy batches with deterministic integer seeds.\"\"\"
    dev = torch.device(device)
    generator = torch.Generator(device=dev.type if dev.type == "cuda" else "cpu").manual_seed(seed)
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator, device=device)"""

assert old_code in code, "old_code not found in examples/_shared/offline_data.py"
code = code.replace(old_code, new_code)

with open("examples/_shared/offline_data.py", "w") as f:
    f.write(code)

print("examples/_shared/offline_data.py fixed successfully!")
