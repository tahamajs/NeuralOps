import torch
from Codes.framework.attentions.bam_umoe import BAMUMoEBlock
class DummyConfig:
    hidden_size = 768
    num_attention_heads = 12
    max_position_embeddings = 512
config = DummyConfig()
model = BAMUMoEBlock(config)
x = torch.randn(8, 128, 768)
mask = torch.ones(8, 1, 1, 128)
try:
    model(x, attention_mask=mask)
    print("BAM_UMOE SUCCESS")
except Exception as e:
    print(f"BAM_UMOE ERROR: {e}")

from Codes.framework.attentions.chela_slay import ChelaSlayAttention
model2 = ChelaSlayAttention(config)
try:
    model2(x, attention_mask=mask)
    print("CHELA_SLAY SUCCESS")
except Exception as e:
    print(f"CHELA_SLAY ERROR: {e}")
