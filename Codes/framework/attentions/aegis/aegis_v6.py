# ------------------------------------------------------------
# Framework integration
# ------------------------------------------------------------
from framework.attentions.core.registry import ATTENTION_REGISTRY

@ATTENTION_REGISTRY.register("aegis_v6")
def inject_aegis_v6(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    cfg = AegisConfigV3(
        hidden_size=model_config.hidden_size,
        num_heads=model_config.num_attention_heads,
        head_dim=model_config.hidden_size // model_config.num_attention_heads,
        window_size=getattr(comp_config, 'block_size', 64),
        svd_rank=getattr(comp_config, 'compression_rank', 128),
        low_rank_update=getattr(comp_config, 'low_rank_update', 16),
        kernel_feature_dim=getattr(comp_config, 'rff_features', 128),
        dropout=getattr(comp_config, 'dropout', 0.1),
        use_query_gate=True,
        use_input_gate=True,
        use_pre_norm=True,
    )
    pretrained = {
        'q': old_attn.q_lin.weight.data.clone().detach(),
        'k': old_attn.k_lin.weight.data.clone().detach(),
        'v': old_attn.v_lin.weight.data.clone().detach(),
        'o': old_attn.out_lin.weight.data.clone().detach()
    }
    new_attn = AegisAttentionV3(cfg, pretrained)
    layer.attention = new_attn
    return new_attn