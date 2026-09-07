# d035 gated_deltanet_block  [attention]
aliases: gated_deltanet_block
file: Codes/framework/attentions/gated_deltanet_block.py  doc: Docs/content/attentions/gated_deltanet_block.md
doc: Gated DeltaNet Block with Optional Multi-Head Attention Bypass. | Implements an efficient hybrid linear-recurrent block switching dynamically between standard | multi-head self-attention and Bidirectional Gated DeltaNet with pretrained weight transfer.
class GatedDeltaNetBlock(BaseAttention)
  doc: Gated DeltaNet Transformer Block with Pretrained Weight Transfer. | Mathematical Formulation:
__init__(config, rank, d_model, num_heads, use_full_attention)
forward(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
features: delta,gating,rank
loc: 218