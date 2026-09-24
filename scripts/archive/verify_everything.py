import os, glob, json, sys, stat

print("=== PART 1: CODE FIXES ===")
checks = {
    'Codes/framework/compression_engine/zerokv_neo_compressor.py': ['out_lin', 'out_proj', 'o_proj', 'c_proj', 'output.dense', 'compressed_layer = base_layer'],
    'Codes/framework/attentions/galerkin.py': ['hasattr(src, "out_lin")', 'logger.debug'],
    'Codes/framework/attentions/flash_attention3.py': ['hasattr(src, "out_lin")', 'logger.debug'],
    'Codes/framework/attentions/flash_attention3_fp8.py': ['hasattr(src, "out_lin")', 'logger.debug'],
    'Codes/framework/attentions/rala.py': ['self.alpha.requires_grad = False'],
    'Codes/framework/attentions/eat.py': ['logger.debug'],
    'Codes/framework/attentions/aegis_v4_module.py': ['F.softmax'],
    'Codes/framework/attentions/aegis_v5_module.py': ['F.softmax'],
    'Codes/framework/attentions/aegis_v6.py': ['min(self.cfg.clamp_max, 8.0)', '8.0'],
    'Codes/framework/attentions/aegis_spectron_fp8.py': ['logger.debug'],
    'Codes/framework/attentions/squeezing_heads.py': ['logger.warning'],
    'Codes/framework/attentions/routing.py': ['scores = torch.matmul'],
    'Codes/framework/attentions/fused_svd_kernel.py': ['fill_diagonal_'],
    'Codes/framework/attentions/compressed_mha.py': ['logger.debug'],
    'Codes/framework/attentions/compressed_attention_layer.py': ['logger.debug', 'not self.is_nested_encoder:'],
    'Codes/framework/attentions/decay_linear_attention.py': ['linspace(2.0, 4.0', 'min=0.85, max=0.999'],
    'Codes/framework/attentions/svd_linear.py': ['192', 'out_lin', 'out_proj', 'o_proj', 'c_proj', 'output.dense'],
    'Codes/framework/attentions/hgrn2_gated_deltanet.py': ['hgrn_ok', 'deltanet_ok'],
    'Codes/framework/attentions/gated_delta_net.py': ['chunk_size: int = 32', '_delta_chunk_forward'],
    'Codes/framework/attentions/gated_deltanet_block.py': ['is_nested_encoder'],
    'Codes/framework/attentions/svd_mamba.py': ['nn.Linear(self.inner_dim', 'nn.Linear(self.dt_rank', 'max=1.0'],
    'Codes/framework/attentions/svd_llm_v2.py': ['192', '0.999'],
    'Codes/framework/attentions/gqa.py': ['kv_repeat = 1'],
    'Codes/framework/attentions/jamba2.py': ['Jamba2Block', 'SVDMambaSub'],
    'Codes/framework/compression_engine/cur_moe_compressor.py': ['visited'],
    'Codes/framework/compression_engine/sdrm_compressor.py': ['out_lin', 'out_proj', 'o_proj', 'c_proj', 'output.dense'],
    'Codes/framework/compression_engine/kronecker_compressor.py': ['0.80'],
    'Codes/framework/compression_engine/a3_compressor.py': ['0.40'],
    'Codes/framework/compression_engine/aasvd_compressor.py': ['0.40'],
    'Codes/framework/compression_engine/simple_ssm.py': ['nn.init.zeros_(self.out_proj.weight)'],
    'Codes/framework/run_quality_sweep.py': ['expandable_segments:True', 'set_per_process_memory_fraction(0.85)', 'target_micro = curr_batch_size', 'args.seed'],
    'Codes/framework/run_agnews_multi_class_sweep.py': ['SKIP_MODULES_BY_MODEL', 'BATCH_SIZE_BY_MODEL', 'RECOVERY_OVERRIDES']
}

for file, markers in checks.items():
    if not os.path.exists(file):
        print(f"MISSING_FILE|{file}")
        continue
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    for m in markers:
        if m in content:
            print(f"OK|{file}|{m}")
        else:
            print(f"MISSING|{file}|{m}")

print("\n=== PART 3: INJECT STUBS ===")
stubs = ['adaptive_span', 'adaptive_sparse', 'adasplash', 'cluster', 'cosformer', 'galerkin', 'hla', 'kda', 'leka_v1', 'leka_v2', 'leka_v3']
for stub in stubs:
    path = f'Codes/framework/attentions/{stub}_inject.py'
    if not os.path.exists(path):
        print(f"MISSING_FILE|{path}")
        continue
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        if f'from framework.attentions.{stub} import *' in content:
            print(f"OK|{path}")
        else:
            print(f"NOT_STUB|{path}")

print("\n=== PART 2: ARTIFACTS ===")
for mandatory in ['tables/main_results.csv', 'tables/rank_ablation.csv', 'metrics/ci95.json', 'env/neurips_checklist.md']:
    # these are global or per module? Prompt says "For every module in papers/modules/<module>/"
    pass

modules = glob.glob('papers/modules/*')
if not modules:
    print("MISSING_ALL|papers/modules is empty")
else:
    for mod in modules:
        for f in ['paper.md', 'citation.bib', 'hf_hub.txt', 'tables/main_results.csv', 'metrics/ci95.json', 'env/neurips_checklist.md']:
            if not os.path.exists(os.path.join(mod, f)):
                print(f"MISSING_ARTIFACT|{mod}|{f}")

print("\n=== PART 4: EXTERNAL BASELINES ===")
if not os.path.exists('tables/external_baselines.csv'):
    print("MISSING|tables/external_baselines.csv")

print("\n=== PART 9: FAILURE TAXONOMY ===")
if not os.path.exists('tables/failures.csv'):
    print("MISSING|tables/failures.csv")
if not os.path.exists('tables/failure_taxonomy.csv'):
    print("MISSING|tables/failure_taxonomy.csv")
