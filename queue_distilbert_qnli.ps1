while (Get-Process -Id 12412 -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 30 }
$out="D:/SVD_PROJECT/results/distilbert_qnli_cached"
New-Item -ItemType Directory -Force -Path $out | Out-Null
& "C:/Users/Hadi Schumacher/anaconda3/python.exe" -u "D:/SVD_PROJECT/Codes/framework/run_quality_sweep.py" --output_dir $out --modules all --models distilbert --datasets qnli --pretrained --max_steps 1 --seq_len 64 --timeout 120 --precision fp16 --batch_size 256 --gradient_checkpointing *> "$out/run.log"
