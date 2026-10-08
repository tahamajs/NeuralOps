"""Tests for ecosystem and integration modules."""
import torch
import torch.nn as nn
from neuralops.integrations.hf_automodel import AutoNeuralOpsModel, register_hf_automodel
from neuralops.integrations.serving import VLLMAttentionAdapter, TGIAttentionAdapter
from neuralops.integrations.distributed_compile import (
    get_deepspeed_config,
    apply_fsdp_wrapping,
    compile_attention_module,
)


def test_hf_automodel():
    assert isinstance(register_hf_automodel(), bool)
    mod = AutoNeuralOpsModel.from_pretrained("dummy_model", attention_type="full_attention")
    assert isinstance(mod, nn.Module)


def test_serving_adapters():
    vllm = VLLMAttentionAdapter(attention_type="linear", head_dim=64, num_heads=4)
    meta = vllm.format_metadata(block_tables=[[1, 2]], context_lens=[10])
    assert meta["max_context_len"] == 10

    tgi = TGIAttentionAdapter(max_batch_size=2)
    assert len(tgi.prepare_batch([1, 2, 3, 4])) == 2


def test_distributed_and_compile():
    ds_cfg = get_deepspeed_config(stage=3)
    assert ds_cfg["zero_optimization"]["stage"] == 3

    lin = nn.Linear(8, 8)
    fsdp_mod = apply_fsdp_wrapping(lin)
    assert isinstance(fsdp_mod, nn.Module)

    comp_mod = compile_attention_module(lin)
    assert isinstance(comp_mod, nn.Module)
