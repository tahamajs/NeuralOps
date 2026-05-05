# Unified Registry for AEGIS Attention variants
from .aegis_hybrid import AegisHybridAttention
from .aegis_v1 import AEGISConfigV1, AEGISAttention as AegisV1Attention
from .aegis_v2 import AegisAttentionConfigV2 as AEGISConfigV2, AegisAttentionV2
from .aegis_v3 import AegisAttention as AegisV3Attention
from .aegis_v4 import AegisAttention as AegisV4Attention
from .aegis_v5 import AegisAttention as AegisV5Attention
from .aegis_v6 import AegisConfigV3 as AegisConfigV6, AegisAttentionV3 as AegisV6Attention

__all__ = [
    'AegisHybridAttention',
    'AEGISConfigV1', 'AegisV1Attention',
    'AEGISConfigV2', 'AegisAttentionV2',
    'AegisV3Attention',
    'AegisV4Attention',
    'AegisV5Attention',
    'AegisConfigV6', 'AegisV6Attention',
]
