$env:HF_HOME = "C:\Users\Hadi Schumacher\.cache\huggingface"
$env:PATH = "C:\Users\Hadi Schumacher\anaconda3;C:\Users\Hadi Schumacher\anaconda3\Scripts;C:\Users\Hadi Schumacher\anaconda3\Library\bin;" + $env:PATH
$modules = @('asvd_attention', 'svd_llm_v2_attention', 'adaptive_svd_attention', 'squeezing_heads', 'gqa_attention', 'primal_attention', 'svd_linear_attention', 'mamba_svd_flash', 'res_svd', 'lora_attention', 'squeezing_heads_distill', 'fusion_attn', 'fused_svd_kernel', 'flash_attention3', 'spectron')
foreach ($m in $modules) {
    & "C:\Users\Hadi Schumacher\anaconda3\python.exe" Codes\framework\eval_perplexity.py --model models\gpt2 --module $m --output_dir results\wikitext2_$m\gpt2\ >> logs\ppl_loop.log 2>&1
}
