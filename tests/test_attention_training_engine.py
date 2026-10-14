"""
Comprehensive test suite verifying all 6 pillars of the post-surgery attention training,
distillation, staged surgery, SVD-PEFT, RL alignment, and stability engine on CUDA.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.trainer.distillation.multi_level_kd import (
    LogitDistillationLoss,
    SpikyFeatureMap,
    HedgehogFeatureMapKDLoss,
    MiniLMValueRelationKDLoss,
    BlockwiseOperatorCloningLoss,
    LinearCKALoss,
)
from neuralops.trainer.surgery_schedules.mohawk import (
    AttentionHomotopyScheduler,
    AlphaBlendingAttentionWrapper,
    MOHAWKTrainer,
)
from neuralops.trainer.peft_recovery.svd_adapters import (
    PiSSALinear,
    DoRALinear,
    inject_pissa_adapters,
)
from neuralops.trainer.rl_alignment.preference_and_onpolicy import (
    SimPOLoss,
    ORPOLoss,
    DynamicRoutingAndRankRL,
)
from neuralops.trainer.regularizers.attention_stability import (
    LearnableAttentionSink,
    GateSaturationPenalty,
    SpectralJacobianRegularizer,
)
from neuralops.trainer.ttt.test_time_trainer import TTTLinearAttentionLayer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for 6-pillar engine")
class TestAttentionTrainingEngine:

    def test_pillar1_multi_level_kd_losses(self):
        B, T, V = 2, 16, 1000
        s_logits = torch.randn(B, T, V, device="cuda", requires_grad=True)
        t_logits = torch.randn(B, T, V, device="cuda")
        labels = torch.randint(0, V, (B, T), device="cuda")

        for mode in ["forward_kl", "reverse_kl", "beta_jsd", "decoupled_kd", "topk_sparse_kd"]:
            loss_fn = LogitDistillationLoss(mode=mode, temperature=2.0)
            loss = loss_fn(s_logits, t_logits, labels=labels)
            assert not torch.isnan(loss) and not torch.isinf(loss)
            loss.backward(retain_graph=True)
            assert s_logits.grad is not None
            s_logits.grad.zero_()

        # Hedgehog Feature Map KD
        H, d = 4, 32
        spiky_q = SpikyFeatureMap(d).cuda()
        spiky_k = SpikyFeatureMap(d).cuda()
        q_s = torch.randn(B, H, T, d, device="cuda", requires_grad=True)
        k_s = torch.randn(B, H, T, d, device="cuda", requires_grad=True)
        phi_q = spiky_q(q_s)
        phi_k = spiky_k(k_s)
        q_t = torch.randn(B, H, T, d, device="cuda")
        k_t = torch.randn(B, H, T, d, device="cuda")

        hh_fn = HedgehogFeatureMapKDLoss(chunk_size=16)
        hh_loss = hh_fn(q_t, k_t, phi_q, phi_k)
        assert not torch.isnan(hh_loss)
        hh_loss.backward()

        # MiniLM Value Relation
        v_s = torch.randn(B, H, T, d, device="cuda", requires_grad=True)
        v_t = torch.randn(B, H, T, d, device="cuda")
        minilm_fn = MiniLMValueRelationKDLoss()
        v_loss = minilm_fn(v_s, v_t)
        assert not torch.isnan(v_loss)
        v_loss.backward()

        # Blockwise Operator Cloning & Linear CKA
        out_s = torch.randn(B, T, 64, device="cuda", requires_grad=True)
        out_t = torch.randn(B, T, 64, device="cuda")
        b_fn = BlockwiseOperatorCloningLoss()
        b_loss = b_fn(out_s, out_t)
        assert not torch.isnan(b_loss)
        b_loss.backward()

        cka_fn = LinearCKALoss()
        cka_loss = cka_fn(out_s, out_t)
        assert not torch.isnan(cka_loss)

    def test_pillar2_staged_surgery_and_homotopy(self):
        # Homotopy scheduler
        sched = AttentionHomotopyScheduler(total_steps=100, schedule="cosine")
        assert abs(sched.get_alpha(0) - 0.0) < 1e-4
        assert abs(sched.get_alpha(100) - 1.0) < 1e-4
        assert 0.0 < sched.get_alpha(50) < 1.0

        # Blending wrapper
        l1 = nn.Linear(32, 32).cuda()
        l2 = nn.Linear(32, 32).cuda()
        wrapper = AlphaBlendingAttentionWrapper(l1, l2)
        x = torch.randn(2, 4, 32, device="cuda")
        
        wrapper.set_alpha(0.0)
        assert torch.allclose(wrapper(x), l1(x))
        wrapper.set_alpha(1.0)
        assert torch.allclose(wrapper(x), l2(x))

    def test_pillar3_pissa_and_dora_adapters(self):
        lin = nn.Linear(64, 128).cuda()
        x = torch.randn(2, 8, 64, device="cuda")
        y_orig = lin(x)

        pissa = PiSSALinear(lin, rank=16).cuda()
        y_pissa = pissa(x)
        # Step-0 exact identity guarantee
        assert torch.allclose(y_orig, y_pissa, atol=1e-4)

        # Gradient flows through adapters
        loss = y_pissa.sum()
        loss.backward()
        assert pissa.lora_A.grad is not None
        assert pissa.lora_B.grad is not None

        # Merge and unload
        merged = pissa.merge_and_unload()
        assert torch.allclose(y_orig, merged(x), atol=1e-4)

        # DoRA test
        dora = DoRALinear(lin, rank=16).cuda()
        y_dora = dora(x)
        assert y_dora.shape == (2, 8, 128)

    def test_pillar4_preference_and_routing_rl(self):
        simpo = SimPOLoss(beta=2.0, gamma=0.5)
        logp_w = torch.tensor([[-0.2, -0.3, -0.1]], device="cuda")
        logp_l = torch.tensor([[-1.5, -2.0, -1.8]], device="cuda")
        loss = simpo(logp_w, logp_l)
        assert not torch.isnan(loss) and loss.item() > 0

        orpo = ORPOLoss(lambda_orpo=0.1)
        sft_loss = torch.tensor(1.2, device="cuda", requires_grad=True)
        loss_orpo = orpo(sft_loss, logp_w, logp_l)
        assert not torch.isnan(loss_orpo)
        loss_orpo.backward()

        routing = DynamicRoutingAndRankRL(num_ranks_or_blocks=4, target_budget_ratio=0.5)
        logits = torch.randn(2, 4, device="cuda", requires_grad=True)
        decisions = routing.sample_routing(logits, tau=1.0)
        penalty = routing.compute_budget_penalty(decisions)
        assert penalty.dim() == 0

    def test_pillar5_stability_regularizers(self):
        sink = LearnableAttentionSink(num_heads=4, head_dim=32, num_sink_tokens=2).cuda()
        k = torch.randn(2, 4, 16, 32, device="cuda")
        v = torch.randn(2, 4, 16, 32, device="cuda")
        k_s, v_s = sink.prepend_sinks(k, v)
        assert k_s.shape == (2, 4, 18, 32)
        assert v_s.shape == (2, 4, 18, 32)

        sat_fn = GateSaturationPenalty(penalty_weight=0.01)
        gates = torch.sigmoid(torch.randn(2, 4, 16, device="cuda"))
        loss_sat = sat_fn(gates)
        assert not torch.isnan(loss_sat)

        spec_fn = SpectralJacobianRegularizer(max_sigma=2.0)
        W = torch.randn(64, 64, device="cuda", requires_grad=True)
        loss_spec = spec_fn(W)
        assert not torch.isnan(loss_spec)

    def test_pillar6_test_time_training_layer(self):
        ttt = TTTLinearAttentionLayer(dim=64, num_heads=2).cuda()
        x = torch.randn(2, 8, 64, device="cuda", requires_grad=True)
        out, _ = ttt(x)
        assert out.shape == (2, 8, 64)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()
