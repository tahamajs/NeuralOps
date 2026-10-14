import pytest
import torch
import torch.nn as nn
from neuralops.attentions.linear.cosformer import CosformerAttention
from neuralops.core.mask_utils import prepare_mask_and_positions, mask_kv_states


def test_prepare_mask_and_positions_logic():
    # Left padded: [PAD, PAD, A, B]
    # Right padded: [A, B, PAD, PAD]
    hidden = torch.randn(2, 4, 16)
    mask = torch.tensor([
        [0, 0, 1, 1],  # left-padded
        [1, 1, 0, 0],  # right-padded
    ], dtype=torch.long)

    valid_mask, pos_ids, clean = prepare_mask_and_positions(hidden, attention_mask=mask)

    # Valid mask matches binary
    assert valid_mask[0].tolist() == [False, False, True, True]
    assert valid_mask[1].tolist() == [True, True, False, False]

    # For valid tokens, position IDs must start from 0
    # Left-padded valid tokens at indices 2 and 3 should have pos_ids [0, 1]
    assert pos_ids[0, 2:].tolist() == [0, 1]
    # Right-padded valid tokens at indices 0 and 1 should have pos_ids [0, 1]
    assert pos_ids[1, :2].tolist() == [0, 1]


def test_cosformer_left_vs_right_padding_equivalence():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attn = CosformerAttention(hidden_size=32, num_heads=2, is_causal=True).to(device).eval()

    # Create 4 identical meaningful tokens
    tokens = torch.randn(1, 4, 32, device=device)

    # Construct Left-padded batch: [PAD, PAD, PAD, PAD, T0, T1, T2, T3] (length 8)
    pad = torch.zeros(1, 4, 32, device=device)
    left_x = torch.cat([pad, tokens], dim=1)
    left_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], device=device)

    # Construct Right-padded batch: [T0, T1, T2, T3, PAD, PAD, PAD, PAD] (length 8)
    right_x = torch.cat([tokens, pad], dim=1)
    right_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], device=device)

    with torch.no_grad():
        left_out, _ = attn(left_x, attention_mask=left_mask)
        right_out, _ = attn(right_x, attention_mask=right_mask)

    # Extract the 4 valid tokens from both batches
    valid_from_left = left_out[:, 4:, :]
    valid_from_right = right_out[:, :4, :]

    max_diff = (valid_from_left - valid_from_right).abs().max().item()
    # Left vs right padded representations for valid tokens must match closely (tolerance 1e-4)
    assert max_diff < 1e-3, f"Left and Right padding diverged for valid tokens! Max diff: {max_diff}"
