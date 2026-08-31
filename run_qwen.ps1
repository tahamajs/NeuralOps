$env:HF_HUB_OFFLINE="1"
$env:TRANSFORMERS_OFFLINE="1"
$out="D:/SVD_PROJECT/results/qwen_rte_fast"
New-Item -ItemType Directory -Force -Path $out | Out-Null
& "C:/Users/Hadi Schumacher/anaconda3/python.exe" -u "D:/SVD_PROJECT/Codes/framework/run_quality_sweep.py" --output_dir $out --modules all --models qwen --datasets rte --pretrained --max_steps 1 --seq_len 64 --timeout 120 --precision fp16 --batch_size 256 *> "$out/run.log"
