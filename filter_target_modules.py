import re

text = """
  ۴. ۳. ۱ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول fusion_attn . . . . . . . . . . . . . ۱۱
  ۴. ۳. ۲ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول primal_attention . . . . . . . . . . . . . . . . ۱۹
  ۴. ۳. ۳ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول gqa_attention . . . . . . . . . . . . . . . . . ۲۷
  ۴. ۳. ۴ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول adaptive_sparse_attention . . . . . . . . . . . . . ۳۵
  ۴. ۳. ۵ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﺎژول adaptive_span_attention . . . . . . . . . . . . . ۴۳
  ۴. ۳. ۶ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول adasplash_attention . . . . . . . . . . . . . . . ۵۱
  ۴. ۳. ۷ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول goldfinch_ahn . . . . . . . . . . . . . . . . . ۵۹
  ۴. ۳. ۸ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول halo_regla . . . . . . . . . . . . . . . . . . . ۶۷
  ۴. ۳. ۹ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول nsa_attention . . . . . . . . . . . . . . . . . . ۷۵
  ۴. ۳. ۱۰ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول nha_attention . . . . . . . . . . . . . . . . . ۸۳
  ۴. ۳. ۱۱ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول squeezing_heads_distiller . . . . . . . . . . . . . ۹۱
  ۴. . ۱۲ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول progressive_token_pruning . . . . . . . . . . . . ۹۸
  ۴. ۳. ۱۳ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول eat_attention . . . . . . . . . . . . . . . . . . ۱۰۵
  ۴. ۳. ۱۴ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول infini_attention . . . . . . . . . . . . . . . . . ۱۱۳
  ۴. ۳. ۱۵ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول optimized_hybrid_attention . . . . . . . . . . . . ۱۲۱
  ۴. ۳. ۱۶ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول asvd_attention . . . . . . . . . . . . . . . . . ۱۲۹
  ۴. ۳. ۱۷ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول rpc_attention . . . . . . . . . . . . . . . . . . ۱۳۷
  ۴. ۳. ۱۸ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول svd_mamba . . . . . . . . . . . . . . . . . . ۱۴۵
  ۴. ۳. ۱۹ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول svd_linear_attention . . . . . . . . . . . . . . . ۱۵۳
  ۴. ۳. ۲۰ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول res_svd . . . . . . . . . . . . . . . . . . . . ۱۶۲
  ۴. ۳. ۲۱ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول minimal_aurora . . . . . . . . . . . . . . . . ۱۶۹
  ۴. ۳. ۲۲ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول minimal_nova . . . . . . . . . . . . . . . . . ۱۷۷
  ۴. ۳. ۲۳ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول minimal_prism . . . . . . . . . . . . . . . . . ۱۸۵
  ۴. ۳. ۲۴ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول minimal_svd . . . . . . . . . . . . . . . . . . ۱۹۳
  ۴. ۳. ۲۵ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول svd_llm_v2_attention . . . . . . . . . . . . . . ۲۰۱
  ۴. ۳. ۲۶ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول mla_attention . . . . . . . . . . . . . . . . . ۲۰۹
  ۴. ۳. ۲۷ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول dobi_attention . . . . . . . . . . . . . . . . . ۲۱۷
  ۴. ۳. ۲۸ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول nystrom_attention . . . . . . . . . . . . . . . . ۲۲۳
  ۴. ۳. ۲۹ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول unified_rope . . . . . . . . . . . . . . . . . . ۲۳۱
  ۴. ۳. ۳۰ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول tt . . . . . . . . . . . . . . . . . . . . . . ۲۳۸
  ۴. ۳. ۳۱ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول hgrn2_gated_deltanet . . . . . . . . . . . . . . ۲۴۵
  ۴. ۳. ۳۲ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول hgrn2_layer . . . . . . . . . . . . . . . . . . ۲۵۳
  ۴. ۳. ۳۳ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول SOAFusedAttention . . . . . . . . . . . . . . . ۲۶۰
  ۴. ۳. ۳۴ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول adaptive_svd_attention . . . . . . . . . . . . . . ۲۶۷
  ۴. ۳. ۳۵ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول bam_umoe . . . . . . . . . . . . . . . . . . ۲۷۵
  ۴. ۳. ۳۶ ﮐﺎﻟﺒﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول block_transformer_hiba . . . . . . . . . . . . . ۲۸۳
  ۴. ۳. ۳۷ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول chela_slay . . . . . . . . . . . . . . . . . . . ۲۹۱
  ۴. ۳. ۳۸ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول cluster_attention . . . . . . . . . . . . . . . . ۲۹۹
  ۴. ۳. ۳۹ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول cosformer . . . . . . . . . . . . . . . . . . . ۳۰۷
  ۴. ۳. ۴۰ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول decay_linear_attention . . . . . . . . . . . . . . ۳۱۵
  ۴. ۳. ۴۱ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول expert_choice_moe . . . . . . . . . . . . . . . ۳۲۳
  ۴. ۳. ۴۲ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول flash_attention3 . . . . . . . . . . . . . . . . ۳۳۱
  ۴. ۳. ۴۳ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول flash_attention3_fp8 . . . . . . . . . . . . . . . ۳۳۹
  ۴. ۳. ۴۴ ﮐﺎﻟﺒﺪﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول galerkin_attention . . . . . . . . . . . . . . . . ۳۴۷
  ۴. ۳. ۴۵ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول gated_delta_net . . . . . . . . . . . . . . . . . ۳۵۵
  ۴. ۳. ۴۶ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣژول gated_deltanet_block . . . . . . . . . . . . . . ۳۶۳
  ۴. ۳. ۴۷ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول griffin_ssd . . . . . . . . . . . . . . . . . . . ۳۷۰
  ۴. ۳. ۴۸ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول hymba_transx . . . . . . . . . . . . . . . . . ۳۷۷
  ۴. ۳. ۴۹ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول jamba_info . . . . . . . . . . . . . . . . . . ۳۸۴
  ۴. ۳. ۵۰ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول vspec . . . . . . . . . . . . . . . . . . . . ۳۹۲
  ۴. ۳. ۵۱ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول leka_v1 . . . . . . . . . . . . . . . . . . . . ۴۰۰
  ۴. ۳. ۵۲ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول leka_v2 . . . . . . . . . . . . . . . . . . . . ۴۰۸
  ۴. ۳. ۵۳ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول leka_v3 . . . . . . . . . . . . . . . . . . . . ۴۱۶
  ۴. ۳. ۵۴ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول linformer_attention . . . . . . . . . . . . . . . ۴۲۴
  ۴. ۳. ۵۵ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول tpa_attention . . . . . . . . . . . . . . . . . . ۴۳۲
  ۴. ۳. ۵۶ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول lora_attention . . . . . . . . . . . . . . . . . ۴۳۹
  ۴. ۳. ۵۷ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول xattention . . . . . . . . . . . . . . . . . . . ۴۴۶
  ۴. ۳. ۵۸ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول mamba . . . . . . . . . . . . . . . . . . . . ۴۵۴
  ۴. ۳. ۵۹ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول mixattention_moska . . . . . . . . . . . . . . . ۴۶۲
  ۴. ۳. ۶۰ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول moe_deltakv_paged . . . . . . . . . . . . . . . ۴۷۰
  ۴. ۳. ۶۱ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول nasc_attention . . . . . . . . . . . . . . . . . ۴۷۷
  ۴. ۳. ۶۲ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول nexus . . . . . . . . . . . . . . . . . . . . ۴۸۴
  ۴. ۳. ۶۳ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول orion . . . . . . . . . . . . . . . . . . . . . ۴۹۲
  ۴. ۳. ۶۴ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول padre . . . . . . . . . . . . . . . . . . . . . ۴۹۹
  ۴. ۳. ۶۵ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول performer_attention . . . . . . . . . . . . . . . ۵۰۶
  ۴. ۳. ۶۶ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول phantom . . . . . . . . . . . . . . . . . . . ۵۱۳
  ۴. ۳. ۶۷ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول projection_factorized_attn . . . . . . . . . . . . ۵۲۰
  ۴. ۳. ۶۸ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول rala_attention . . . . . . . . . . . . . . . . . ۵۲۵
  ۴. ۳. ۶۹ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول reap_moe . . . . . . . . . . . . . . . . . . . ۵۳۲
  ۴. ۳. ۷۰ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول regla_attention . . . . . . . . . . . . . . . . . ۵۳۹
  ۴. ۳. ۷۱ ﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول routing_attention . . . . . . . . . . . . . . . . ۵۴۷
  ۴. ۳. ۷۲ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول salt . . . . . . . . . . . . . . . . . . . . . ۵۵۴
  ۴. ۳. ۷۳ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول simple_ssm . . . . . . . . . . . . . . . . . . ۵۵۹
  ۴. ۳. ۷۴ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول span_expanded_attention . . . . . . . . . . . . . ۵۶۶
  ۴. ۳. ۷۵ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول svd_projection_attention . . . . . . . . . . . . . ۵۷۳
  ۴. ۳. ۷۶ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول template_attention . . . . . . . . . . . . . . . ۵۸۰
  ۴. ۳. ۷۷ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول token_compressed_attention . . . . . . . . . . . . ۵۸۷
  ۴. ۳. ۷۸ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول moba . . . . . . . . . . . . . . . . . . . . . ۵۹۴
  ۴. ۳. ۷۹ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول spectra_v2 . . . . . . . . . . . . . . . . . . ۵۹۹
  ۴. ۳. ۸۰ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول compressed_mha . . . . . . . . . . . . . . . . ۶۰۶
  ۴. ۳. ۸۱ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول svd_compressed . . . . . . . . . . . . . . . . ۶۱۴
  ۴. ۳. ۸۲ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول spectron . . . . . . . . . . . . . . . . . . . ۶۲۳
  ۴. ۳. ۸۳ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول soa_module . . . . . . . . . . . . . . . . . . ۶۳۰
  ۴. ۳. ۸۴ ﮐﺎﻟﺒﺪﺷﮑﺎﻓﯽ دﻗﯿﻖ ﻣﺎژول aegis_v5 . . . . . . . . . . . . . . . . . . . ۶۳۶
"""

import os, glob

raw_names = []
for line in text.strip().split("\n"):
    m = re.search(r"ﻣﺎژول\s+([a-zA-Z0-9_]+)", line)
    if not m:
        m = re.search(r"ﺎژول\s+([a-zA-Z0-9_]+)", line)
    if not m:
        m = re.search(r"ﻣژول\s+([a-zA-Z0-9_]+)", line)
    if m:
        raw_names.append(m.group(1).lower())

print(f"Extracted {len(raw_names)} modules from list.")

# Compare against existing runners in Codes/runners
runner_files = glob.glob("/Users/tahamajs/Documents/uni/Project/Codes/runners/run_all_4_*.py")
runner_modules = {os.path.basename(f)[10:-3].lower(): os.path.basename(f)[10:-3] for f in runner_files}

matched = []
unmatched = []

for r in raw_names:
    if r in runner_modules:
        matched.append(runner_modules[r])
    else:
        # Try alternate matches (e.g. without _attention suffix or with it)
        cand = None
        if r.endswith("_attention") and r[:-10] in runner_modules:
            cand = runner_modules[r[:-10]]
        elif (r + "_attention") in runner_modules:
            cand = runner_modules[r + "_attention"]
        elif r == "flash_attention3" and "flash3" in runner_modules:
            cand = runner_modules["flash3"]
        elif r == "flash_attention3_fp8" and "flash3_fp8" in runner_modules:
            cand = runner_modules["flash3_fp8"]
        elif r == "soafusedattention" and "soafusedattention" in runner_modules:
            cand = runner_modules["soafusedattention"]
        
        if cand:
            matched.append(cand)
        else:
            unmatched.append(r)

print(f"Matched: {len(matched)}")
if unmatched:
    print(f"Unmatched: {unmatched}")

# Remove duplicates while preserving order
final_list = []
for m in matched:
    if m not in final_list:
        final_list.append(m)

print(f"Final unique modules to run: {len(final_list)}")
print(",".join(final_list))
